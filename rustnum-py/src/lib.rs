//! Python bindings for `rustnum`: an `ndarray`-like class over `Array<f64>`, `Array<f32>`,
//! `Array<i64>` and `Array<bool>`, with NumPy-style creation functions, operators, reductions,
//! `matmul`, `linalg` and `random`.
//!
//! `numpy.asarray(x)` wraps the array through `__array_interface__` without copying (the NumPy
//! view is read-only and keeps `x` alive). The buffer protocol is not offered: PyO3 only supports
//! `__getbuffer__` for the stable ABI from Python 3.11 on, and this crate targets `abi3-py39`.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};
use rustnum as np;
use rustnum::{Array, SliceArg};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

// ---------------------------------------------------------------------------------------
// dtypes and data
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum DType {
    Bool,
    I64,
    F32,
    F64,
}

impl DType {
    fn parse(s: &str) -> PyResult<DType> {
        Ok(match s {
            "float64" | "float" | "f8" | "double" | "<f8" => DType::F64,
            "float32" | "f4" | "single" | "<f4" => DType::F32,
            "int64" | "int" | "i8" | "<i8" | "intp" => DType::I64,
            "bool" | "bool_" | "?" | "|b1" => DType::Bool,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "data type '{other}' not understood"
                )));
            }
        })
    }
    fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I64 => "int64",
            DType::F32 => "float32",
            DType::F64 => "float64",
        }
    }
    fn typestr(self) -> &'static str {
        match self {
            DType::Bool => "|b1",
            DType::I64 => "<i8",
            DType::F32 => "<f4",
            DType::F64 => "<f8",
        }
    }
}

/// Promotion of two array dtypes, following NumPy (`int64 + float32 -> float64`).
fn promote(a: DType, b: DType) -> DType {
    match (a, b) {
        (DType::I64, DType::F32) | (DType::F32, DType::I64) => DType::F64,
        _ => a.max(b),
    }
}

#[derive(Clone)]
enum Data {
    Bool(Array<bool>),
    I64(Array<i64>),
    F32(Array<f32>),
    F64(Array<f64>),
}

macro_rules! each {
    ($d:expr, $a:ident => $e:expr) => {
        match $d {
            Data::Bool($a) => $e,
            Data::I64($a) => $e,
            Data::F32($a) => $e,
            Data::F64($a) => $e,
        }
    };
}

/// Like `each!` but rewraps the result in the same variant.
macro_rules! each_same {
    ($d:expr, $a:ident => $e:expr) => {
        match $d {
            Data::Bool($a) => Data::Bool($e),
            Data::I64($a) => Data::I64($e),
            Data::F32($a) => Data::F32($e),
            Data::F64($a) => Data::F64($e),
        }
    };
}

impl Data {
    fn dtype(&self) -> DType {
        match self {
            Data::Bool(_) => DType::Bool,
            Data::I64(_) => DType::I64,
            Data::F32(_) => DType::F32,
            Data::F64(_) => DType::F64,
        }
    }
    fn shape(&self) -> Vec<usize> {
        each!(self, a => a.shape().to_vec())
    }
    fn cast(&self, to: DType) -> Data {
        if self.dtype() == to {
            return self.clone();
        }
        each!(self, a => match to {
            DType::Bool => Data::Bool(a.astype::<bool>()),
            DType::I64 => Data::I64(a.astype::<i64>()),
            DType::F32 => Data::F32(a.astype::<f32>()),
            DType::F64 => Data::F64(a.astype::<f64>()),
        })
    }
}

/// Runs `f`, turning a Rust panic (NumPy-style misuse message) into a `ValueError`.
fn guard<R>(f: impl FnOnce() -> R) -> PyResult<R> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "rustnum panicked".to_string());
        if msg.contains("out of bounds") {
            PyIndexError::new_err(msg)
        } else {
            PyValueError::new_err(msg)
        }
    })
}

fn np_err(e: np::Error) -> PyErr {
    match e {
        np::Error::Singular | np::Error::NotPositiveDefinite | np::Error::NoConvergence(_) => {
            PyValueError::new_err(e.to_string())
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

// ---------------------------------------------------------------------------------------
// Scalars and conversion from Python objects
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Scalar {
    fn from_py(o: &Bound<'_, PyAny>) -> Option<Scalar> {
        if o.is_instance_of::<pyo3::types::PyBool>() {
            return o.extract::<bool>().ok().map(Scalar::Bool);
        }
        if o.is_instance_of::<pyo3::types::PyInt>() {
            return o.extract::<i64>().ok().map(Scalar::Int);
        }
        if o.is_instance_of::<pyo3::types::PyFloat>() {
            return o.extract::<f64>().ok().map(Scalar::Float);
        }
        // numpy scalars and other numbers
        if o.hasattr("__index__").unwrap_or(false) && !o.hasattr("__float__").unwrap_or(false) {
            return o.extract::<i64>().ok().map(Scalar::Int);
        }
        if o.hasattr("dtype").unwrap_or(false)
            && let Ok(f) = o.extract::<f64>()
        {
            let kind = o
                .getattr("dtype")
                .and_then(|d| d.getattr("kind"))
                .and_then(|k| k.extract::<String>())
                .unwrap_or_default();
            return Some(match kind.as_str() {
                "b" => Scalar::Bool(f != 0.0),
                "i" | "u" => Scalar::Int(f as i64),
                _ => Scalar::Float(f),
            });
        }
        None
    }
    fn dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::Float(_) => DType::F64,
        }
    }
    fn as_f64(self) -> f64 {
        match self {
            Scalar::Bool(b) => f64::from(u8::from(b)),
            Scalar::Int(i) => i as f64,
            Scalar::Float(f) => f,
        }
    }
    fn as_i64(self) -> i64 {
        match self {
            Scalar::Bool(b) => i64::from(b),
            Scalar::Int(i) => i,
            Scalar::Float(f) => f as i64,
        }
    }
}

fn scalar_f32(x: f32) -> Scalar {
    Scalar::Float(f64::from(x))
}

fn scalar_to_py(py: Python<'_>, s: Scalar) -> PyResult<Py<PyAny>> {
    Ok(match s {
        Scalar::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        Scalar::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        Scalar::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
    })
}

/// Flattens nested Python sequences into (shape, scalars).
fn flatten(
    o: &Bound<'_, PyAny>,
    shape: &mut Vec<usize>,
    depth: usize,
    out: &mut Vec<Scalar>,
) -> PyResult<()> {
    if let Ok(arr) = o.extract::<PyRef<'_, PyArray>>() {
        let d = arr.data.shape();
        for (i, n) in d.iter().enumerate() {
            set_dim(shape, depth + i, *n)?;
        }
        out.extend(arr.scalars());
        return Ok(());
    }
    if let Some(s) = Scalar::from_py(o) {
        if shape.len() > depth {
            return Err(PyValueError::new_err(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape",
            ));
        }
        out.push(s);
        return Ok(());
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(l) = o.cast::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = o.cast::<PyTuple>() {
        t.iter().collect()
    } else if o.hasattr("tolist").unwrap_or(false) {
        let l = o.call_method0("tolist")?;
        return flatten(&l, shape, depth, out);
    } else {
        return Err(PyTypeError::new_err(format!(
            "cannot convert object of type '{}' to an array",
            o.get_type().name()?
        )));
    };
    set_dim(shape, depth, items.len())?;
    let before = out.len();
    for item in &items {
        flatten(item, shape, depth + 1, out)?;
    }
    if items.is_empty() {
        let _ = before;
    }
    Ok(())
}

fn set_dim(shape: &mut Vec<usize>, depth: usize, n: usize) -> PyResult<()> {
    if shape.len() == depth {
        shape.push(n);
        Ok(())
    } else if shape[depth] != n {
        Err(PyValueError::new_err(
            "setting an array element with a sequence. The requested array has an inhomogeneous shape",
        ))
    } else {
        Ok(())
    }
}

fn build(scalars: &[Scalar], shape: &[usize], dtype: Option<DType>) -> PyResult<Data> {
    let want = dtype.unwrap_or_else(|| {
        if scalars.is_empty() {
            DType::F64
        } else {
            scalars.iter().map(|s| s.dtype()).max().unwrap()
        }
    });
    let size: usize = shape.iter().product();
    if size != scalars.len() {
        return Err(PyValueError::new_err(
            "setting an array element with a sequence. The requested array has an inhomogeneous shape",
        ));
    }
    Ok(match want {
        DType::Bool => Data::Bool(Array::from_vec(
            scalars.iter().map(|s| s.as_f64() != 0.0).collect(),
            shape,
        )),
        DType::I64 => Data::I64(Array::from_vec(
            scalars.iter().map(|s| s.as_i64()).collect(),
            shape,
        )),
        DType::F32 => Data::F32(Array::from_vec(
            scalars.iter().map(|s| s.as_f64() as f32).collect(),
            shape,
        )),
        DType::F64 => Data::F64(Array::from_vec(
            scalars.iter().map(|s| s.as_f64()).collect(),
            shape,
        )),
    })
}

fn parse_dtype(dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Option<DType>> {
    match dtype {
        None => Ok(None),
        Some(d) if d.is_none() => Ok(None),
        Some(d) => {
            if let Ok(s) = d.extract::<String>() {
                return DType::parse(&s).map(Some);
            }
            if d.is(d.py().get_type::<pyo3::types::PyFloat>()) {
                return Ok(Some(DType::F64));
            }
            if d.is(d.py().get_type::<pyo3::types::PyBool>()) {
                return Ok(Some(DType::Bool));
            }
            if d.is(d.py().get_type::<pyo3::types::PyInt>()) {
                return Ok(Some(DType::I64));
            }
            DType::parse(&d.str()?.to_string()).map(Some)
        }
    }
}

fn parse_shape(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(n) = shape.extract::<usize>() {
        return Ok(vec![n]);
    }
    shape.extract::<Vec<usize>>()
}

// ---------------------------------------------------------------------------------------
// The ndarray class
// ---------------------------------------------------------------------------------------

/// A NumPy-like n-dimensional array backed by rustnum.
#[pyclass(name = "ndarray", module = "rustnum_py")]
#[derive(Clone)]
struct PyArray {
    data: Data,
}

impl PyArray {
    fn new(data: Data) -> Self {
        PyArray { data }
    }
    fn scalars(&self) -> Vec<Scalar> {
        match &self.data {
            Data::Bool(a) => a.iter().map(Scalar::Bool).collect(),
            Data::I64(a) => a.iter().map(Scalar::Int).collect(),
            Data::F32(a) => a.iter().map(|x| Scalar::Float(f64::from(x))).collect(),
            Data::F64(a) => a.iter().map(Scalar::Float).collect(),
        }
    }
    fn scalar_at(&self, index: &[usize]) -> Scalar {
        match &self.data {
            Data::Bool(a) => Scalar::Bool(a.get(index)),
            Data::I64(a) => Scalar::Int(a.get(index)),
            Data::F32(a) => Scalar::Float(f64::from(a.get(index))),
            Data::F64(a) => Scalar::Float(a.get(index)),
        }
    }
    fn make_contiguous(&mut self) {
        if each!(&self.data, a => a.is_contiguous()) {
            return;
        }
        self.data = each_same!(&self.data, a => a.to_contiguous());
    }
}

/// Either an array or a Python scalar, the right-hand side of a binary operator.
enum Operand {
    Array(Data),
    Scalar(Scalar),
}

fn operand(o: &Bound<'_, PyAny>) -> PyResult<Operand> {
    if let Ok(a) = o.extract::<PyRef<'_, PyArray>>() {
        return Ok(Operand::Array(a.data.clone()));
    }
    if let Some(s) = Scalar::from_py(o) {
        return Ok(Operand::Scalar(s));
    }
    if o.hasattr("tolist").unwrap_or(false)
        || o.cast::<PyList>().is_ok()
        || o.cast::<PyTuple>().is_ok()
    {
        let mut shape = vec![];
        let mut out = vec![];
        flatten(o, &mut shape, 0, &mut out)?;
        return Ok(Operand::Array(build(&out, &shape, None)?));
    }
    Err(PyTypeError::new_err(format!(
        "unsupported operand type '{}'",
        o.get_type().name()?
    )))
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    Max,
    Min,
}

fn target_dtype(op: Op, a: DType, b: &Operand) -> DType {
    let bd = match b {
        Operand::Array(d) => d.dtype(),
        Operand::Scalar(s) => {
            // weak Python scalars: a float scalar only widens integer/bool arrays
            match (s.dtype(), a) {
                (DType::F64, DType::F32 | DType::F64) => a,
                (DType::F64, _) => DType::F64,
                (DType::I64, DType::Bool) => DType::I64,
                (DType::I64, other) => other,
                (DType::Bool, other) => other,
                (x, _) => x,
            }
        }
    };
    let mut t = match b {
        Operand::Array(_) => promote(a, bd),
        Operand::Scalar(_) => promote(a, bd),
    };
    if op == Op::Div && t < DType::F32 {
        t = DType::F64;
    }
    t
}

macro_rules! arith {
    ($op:expr, $x:expr, $y:expr, $T:ty) => {{
        let (x, y): (&Array<$T>, &Array<$T>) = ($x, $y);
        match $op {
            Op::Add => Ok(x + y),
            Op::Sub => Ok(x - y),
            Op::Mul => Ok(x * y),
            Op::Div => Ok(x / y),
            Op::Pow => Ok(np::power(x, y)),
            Op::Max => Ok(np::maximum(x, y)),
            Op::Min => Ok(np::minimum(x, y)),
            _ => Err(()),
        }
    }};
}

macro_rules! compare {
    ($op:expr, $x:expr, $y:expr) => {{
        let (x, y) = ($x, $y);
        match $op {
            Op::Lt => x.lt(y),
            Op::Le => x.le(y),
            Op::Gt => x.gt(y),
            Op::Ge => x.ge(y),
            Op::Eq => x.eq_elem(y),
            _ => x.ne_elem(y),
        }
    }};
}

fn scalar_data(s: Scalar, to: DType) -> Data {
    match to {
        DType::Bool => Data::Bool(Array::scalar(s.as_f64() != 0.0)),
        DType::I64 => Data::I64(Array::scalar(s.as_i64())),
        DType::F32 => Data::F32(Array::scalar(s.as_f64() as f32)),
        DType::F64 => Data::F64(Array::scalar(s.as_f64())),
    }
}

fn binary(op: Op, a: &Data, b: Operand, swapped: bool) -> PyResult<Data> {
    let t = target_dtype(op, a.dtype(), &b);
    let left = a.cast(t);
    let right = match b {
        Operand::Array(d) => d.cast(t),
        Operand::Scalar(s) => scalar_data(s, t),
    };
    let (l, r) = if swapped {
        (right, left)
    } else {
        (left, right)
    };
    guard(|| match (&l, &r) {
        (Data::F64(x), Data::F64(y)) => cmp_or_arith::<f64>(op, x, y).map(|r| r.into_f64()),
        (Data::F32(x), Data::F32(y)) => cmp_or_arith::<f32>(op, x, y).map(|r| r.into_f32()),
        (Data::I64(x), Data::I64(y)) => cmp_or_arith::<i64>(op, x, y).map(|r| r.into_i64()),
        (Data::Bool(x), Data::Bool(y)) => match op {
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                Ok(Data::Bool(compare!(op, x, y)))
            }
            Op::Add | Op::Max => Ok(Data::Bool(np::logical_or(x, y))),
            Op::Mul | Op::Min => Ok(Data::Bool(np::logical_and(x, y))),
            _ => Err("numpy boolean subtract, the `-` operator, is not supported".to_string()),
        },
        _ => unreachable!("operands were promoted to one dtype"),
    })?
    .map_err(PyTypeError::new_err)
}

enum Res<T: np::Element> {
    Num(Array<T>),
    Bool(Array<bool>),
}
impl Res<f64> {
    fn into_f64(self) -> Data {
        match self {
            Res::Num(a) => Data::F64(a),
            Res::Bool(a) => Data::Bool(a),
        }
    }
}
impl Res<f32> {
    fn into_f32(self) -> Data {
        match self {
            Res::Num(a) => Data::F32(a),
            Res::Bool(a) => Data::Bool(a),
        }
    }
}
impl Res<i64> {
    fn into_i64(self) -> Data {
        match self {
            Res::Num(a) => Data::I64(a),
            Res::Bool(a) => Data::Bool(a),
        }
    }
}

fn cmp_or_arith<T: np::Num>(op: Op, x: &Array<T>, y: &Array<T>) -> Result<Res<T>, String> {
    match op {
        Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne => Ok(Res::Bool(compare!(op, x, y))),
        _ => arith!(op, x, y, T)
            .map(Res::Num)
            .map_err(|()| "unsupported operation".to_string()),
    }
}

fn float_ops_only(d: &Data) -> PyResult<()> {
    match d {
        Data::F64(_) | Data::F32(_) => Ok(()),
        _ => Err(PyTypeError::new_err(
            "operation requires a floating point array",
        )),
    }
}

fn axis_arg(axis: Option<isize>) -> Option<isize> {
    axis
}

/// Result of a reduction: a Python scalar for `axis=None`, an array otherwise.
fn reduction_result(
    py: Python<'_>,
    whole: Option<Scalar>,
    arr: Option<Data>,
) -> PyResult<Py<PyAny>> {
    if let Some(s) = whole {
        return scalar_to_py(py, s);
    }
    Ok(Py::new(py, PyArray::new(arr.unwrap()))?.into_any())
}

fn index_scalar(d: &Data, flat: usize) -> Scalar {
    match d {
        Data::Bool(a) => Scalar::Bool(a.iter().nth(flat).unwrap()),
        Data::I64(a) => Scalar::Int(a.iter().nth(flat).unwrap()),
        Data::F32(a) => Scalar::Float(f64::from(a.iter().nth(flat).unwrap())),
        Data::F64(a) => Scalar::Float(a.iter().nth(flat).unwrap()),
    }
}

#[pymethods]
impl PyArray {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.data.shape())
    }
    #[getter]
    fn ndim(&self) -> usize {
        self.data.shape().len()
    }
    #[getter]
    fn size(&self) -> usize {
        each!(&self.data, a => a.size())
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.data.dtype().name()
    }
    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> PyArray {
        PyArray::new(each_same!(&self.data, a => a.t()))
    }
    #[getter]
    fn itemsize(&self) -> usize {
        match self.data.dtype() {
            DType::Bool => 1,
            DType::F32 => 4,
            _ => 8,
        }
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.itemsize() * self.size()
    }

    /// NumPy array interface: lets `numpy.asarray(x)` view this array without copying.
    #[getter]
    fn __array_interface__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let mut this = slf.borrow_mut();
        this.make_contiguous();
        let (ptr, shape) = each!(&this.data, a => (
            a.as_slice().map_or(0usize, |s| s.as_ptr() as usize),
            a.shape().to_vec()
        ));
        let d = PyDict::new(py);
        d.set_item("shape", PyTuple::new(py, shape)?)?;
        d.set_item("typestr", this.data.dtype().typestr())?;
        d.set_item("data", (ptr, true))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.data
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of unsized object"))
    }
    fn __repr__(&self) -> String {
        each!(&self.data, a => format!("{a:?}"))
    }
    fn __str__(&self) -> String {
        each!(&self.data, a => a.to_string())
    }
    fn __bool__(&self) -> PyResult<bool> {
        if self.size() != 1 {
            return Err(PyValueError::new_err(
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
            ));
        }
        Ok(index_scalar(&self.data, 0).as_f64() != 0.0)
    }
    fn __float__(&self) -> PyResult<f64> {
        if self.size() != 1 {
            return Err(PyTypeError::new_err(
                "only length-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(index_scalar(&self.data, 0).as_f64())
    }
    fn __int__(&self) -> PyResult<i64> {
        if self.size() != 1 {
            return Err(PyTypeError::new_err(
                "only length-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(index_scalar(&self.data, 0).as_i64())
    }

    // -- operators ---------------------------------------------------------------------
    fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Add,
            &self.data,
            operand(o)?,
            false,
        )?))
    }
    fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Add,
            &self.data,
            operand(o)?,
            true,
        )?))
    }
    fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Sub,
            &self.data,
            operand(o)?,
            false,
        )?))
    }
    fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Sub,
            &self.data,
            operand(o)?,
            true,
        )?))
    }
    fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Mul,
            &self.data,
            operand(o)?,
            false,
        )?))
    }
    fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Mul,
            &self.data,
            operand(o)?,
            true,
        )?))
    }
    fn __truediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Div,
            &self.data,
            operand(o)?,
            false,
        )?))
    }
    fn __rtruediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Div,
            &self.data,
            operand(o)?,
            true,
        )?))
    }
    fn __pow__(&self, o: &Bound<'_, PyAny>, _m: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Pow,
            &self.data,
            operand(o)?,
            false,
        )?))
    }
    fn __rpow__(&self, o: &Bound<'_, PyAny>, _m: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
        Ok(PyArray::new(binary(
            Op::Pow,
            &self.data,
            operand(o)?,
            true,
        )?))
    }
    fn __neg__(&self) -> PyResult<PyArray> {
        let d = match &self.data {
            Data::F64(a) => Data::F64(-a),
            Data::F32(a) => Data::F32(-a),
            Data::I64(a) => Data::I64(-a),
            Data::Bool(_) => {
                return Err(PyTypeError::new_err(
                    "The numpy boolean negative, the `-` operator, is not supported",
                ));
            }
        };
        Ok(PyArray::new(d))
    }
    fn __abs__(&self) -> PyResult<PyArray> {
        Ok(PyArray::new(match &self.data {
            Data::F64(a) => Data::F64(a.abs()),
            Data::F32(a) => Data::F32(a.abs()),
            Data::I64(a) => Data::I64(a.abs()),
            Data::Bool(a) => Data::Bool(a.clone()),
        }))
    }
    fn __invert__(&self) -> PyResult<PyArray> {
        match &self.data {
            Data::Bool(a) => Ok(PyArray::new(Data::Bool(!a))),
            _ => Err(PyTypeError::new_err(
                "ufunc 'invert' not supported for this dtype",
            )),
        }
    }
    fn __and__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        bool_op(self, o, |x, y| np::logical_and(x, y))
    }
    fn __or__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        bool_op(self, o, |x, y| np::logical_or(x, y))
    }
    fn __xor__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        bool_op(self, o, |x, y| np::logical_xor(x, y))
    }
    fn __matmul__(&self, o: &PyArray) -> PyResult<PyArray> {
        matmul(self, o)
    }
    fn __richcmp__(&self, o: &Bound<'_, PyAny>, op: pyo3::basic::CompareOp) -> PyResult<PyArray> {
        use pyo3::basic::CompareOp::*;
        let op = match op {
            Lt => Op::Lt,
            Le => Op::Le,
            Gt => Op::Gt,
            Ge => Op::Ge,
            Eq => Op::Eq,
            Ne => Op::Ne,
        };
        Ok(PyArray::new(binary(op, &self.data, operand(o)?, false)?))
    }
    fn __iadd__(&mut self, a: &Bound<'_, PyAny>) -> PyResult<()> {
        self.data = binary(Op::Add, &self.data, operand(a)?, false)?;
        Ok(())
    }

    // -- indexing ---------------------------------------------------------------------
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let (args, all_int) = parse_key(&self.data.shape(), key)?;
        let out = guard(|| each_same!(&self.data, a => a.slice(&args)))?;
        let out = PyArray::new(out);
        if all_int && out.ndim() == 0 {
            return scalar_to_py(py, out.scalar_at(&[]));
        }
        Ok(Py::new(py, out)?.into_any())
    }
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (args, _) = parse_key(&self.data.shape(), key)?;
        let v = operand(value)?;
        let dt = self.data.dtype();
        let vd = match v {
            Operand::Array(d) => d.cast(dt),
            Operand::Scalar(s) => scalar_data(s, dt),
        };
        guard(|| match (&mut self.data, vd) {
            (Data::F64(a), Data::F64(v)) => a.slice_assign(&args, v),
            (Data::F32(a), Data::F32(v)) => a.slice_assign(&args, v),
            (Data::I64(a), Data::I64(v)) => a.slice_assign(&args, v),
            (Data::Bool(a), Data::Bool(v)) => a.slice_assign(&args, v),
            _ => unreachable!(),
        })
    }

    // -- shape ----------------------------------------------------------------------
    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<PyArray> {
        let dims: Vec<isize> = if shape.len() == 1 {
            let first = shape.get_item(0)?;
            first
                .extract::<isize>()
                .map(|n| vec![n])
                .or_else(|_| first.extract::<Vec<isize>>())?
        } else {
            shape.extract()?
        };
        let out = guard(|| each_same!(&self.data, a => a.reshape_infer(&dims)))?;
        Ok(PyArray::new(out))
    }
    #[pyo3(signature = (*axes))]
    fn transpose(&self, axes: &Bound<'_, PyTuple>) -> PyResult<PyArray> {
        if axes.is_empty() {
            return Ok(self.T());
        }
        let ax: Vec<usize> = if axes.len() == 1 {
            axes.get_item(0)?.extract()?
        } else {
            axes.extract()?
        };
        let out = guard(|| each_same!(&self.data, a => a.transpose(&ax)))?;
        Ok(PyArray::new(out))
    }
    fn ravel(&self) -> PyArray {
        PyArray::new(each_same!(&self.data, a => a.ravel()))
    }
    fn flatten(&self) -> PyArray {
        PyArray::new(each_same!(&self.data, a => a.flatten()))
    }
    fn copy(&self) -> PyArray {
        PyArray::new(each_same!(&self.data, a => a.copy()))
    }
    fn squeeze(&self) -> PyArray {
        PyArray::new(each_same!(&self.data, a => a.squeeze()))
    }
    fn astype(&self, dtype: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        let t = parse_dtype(Some(dtype))?.ok_or_else(|| PyTypeError::new_err("dtype required"))?;
        Ok(PyArray::new(self.data.cast(t)))
    }
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let shape = self.data.shape();
        let flat = self.scalars();
        nested(py, &flat, &shape, &mut 0)
    }
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.size() != 1 {
            return Err(PyValueError::new_err(
                "can only convert an array of size 1 to a Python scalar",
            ));
        }
        scalar_to_py(py, index_scalar(&self.data, 0))
    }

    // -- reductions -------------------------------------------------------------------
    #[pyo3(signature = (axis=None))]
    fn sum(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Sum)
    }
    #[pyo3(signature = (axis=None))]
    fn prod(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Prod)
    }
    #[pyo3(signature = (axis=None))]
    fn mean(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Mean)
    }
    #[pyo3(signature = (axis=None))]
    fn std(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Std)
    }
    #[pyo3(signature = (axis=None))]
    fn var(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Var)
    }
    #[pyo3(signature = (axis=None))]
    fn min(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Min)
    }
    #[pyo3(signature = (axis=None))]
    fn max(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::Max)
    }
    #[pyo3(signature = (axis=None))]
    fn argmin(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::ArgMin)
    }
    #[pyo3(signature = (axis=None))]
    fn argmax(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        self.reduce(py, axis, Red::ArgMax)
    }
    #[pyo3(signature = (axis=None))]
    fn cumsum(&self, axis: Option<isize>) -> PyResult<PyArray> {
        let d = self.data.cast(if self.data.dtype() == DType::Bool {
            DType::I64
        } else {
            self.data.dtype()
        });
        let out = guard(|| match (&d, axis) {
            (Data::F64(a), None) => Data::F64(a.cumsum()),
            (Data::F32(a), None) => Data::F32(a.cumsum()),
            (Data::I64(a), None) => Data::I64(a.cumsum()),
            (Data::F64(a), Some(ax)) => Data::F64(a.cumsum_axis(ax)),
            (Data::F32(a), Some(ax)) => Data::F32(a.cumsum_axis(ax)),
            (Data::I64(a), Some(ax)) => Data::I64(a.cumsum_axis(ax)),
            _ => unreachable!(),
        })?;
        Ok(PyArray::new(out))
    }
    #[pyo3(signature = (axis=None))]
    fn any(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        let b = self.data.cast(DType::Bool);
        let Data::Bool(a) = b else { unreachable!() };
        match axis {
            None => scalar_to_py(py, Scalar::Bool(a.any())),
            Some(ax) => {
                Ok(Py::new(py, PyArray::new(Data::Bool(guard(|| a.any_axis(ax))?)))?.into_any())
            }
        }
    }
    #[pyo3(signature = (axis=None))]
    fn all(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        let b = self.data.cast(DType::Bool);
        let Data::Bool(a) = b else { unreachable!() };
        match axis {
            None => scalar_to_py(py, Scalar::Bool(a.all())),
            Some(ax) => {
                Ok(Py::new(py, PyArray::new(Data::Bool(guard(|| a.all_axis(ax))?)))?.into_any())
            }
        }
    }
    fn dot(&self, o: &PyArray) -> PyResult<PyArray> {
        dot(self, o)
    }
    fn clip(&self, lo: f64, hi: f64) -> PyResult<PyArray> {
        Ok(PyArray::new(match &self.data {
            Data::F64(a) => Data::F64(a.clip(lo, hi)),
            Data::F32(a) => Data::F32(a.clip(lo as f32, hi as f32)),
            Data::I64(a) => Data::I64(a.clip(lo as i64, hi as i64)),
            Data::Bool(_) => return Err(PyTypeError::new_err("clip does not support bool")),
        }))
    }
}

#[derive(Clone, Copy)]
enum Red {
    Sum,
    Prod,
    Mean,
    Std,
    Var,
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl PyArray {
    fn reduce(&self, py: Python<'_>, axis: Option<isize>, red: Red) -> PyResult<Py<PyAny>> {
        let axis = axis_arg(axis);
        // booleans reduce as int64, integers as float64 for mean-like reductions
        let base = match (self.data.dtype(), red) {
            (DType::Bool, Red::Min | Red::Max | Red::ArgMin | Red::ArgMax) => DType::Bool,
            (DType::Bool, _) => DType::I64,
            (d, _) => d,
        };
        let d = match red {
            Red::Mean | Red::Std | Red::Var if base < DType::F32 => self.data.cast(DType::F64),
            _ => self.data.cast(base),
        };
        if matches!(red, Red::Min | Red::Max | Red::ArgMin | Red::ArgMax) && self.size() == 0 {
            return Err(PyValueError::new_err(
                "zero-size array to reduction operation which has no identity",
            ));
        }
        let (whole, arr) = guard(|| -> PyResult<(Option<Scalar>, Option<Data>)> {
            macro_rules! red_num {
                ($a:expr, $S:path, $D:path) => {{
                    let a = $a;
                    match (red, axis) {
                        (Red::Sum, None) => (Some($S(a.sum())), None),
                        (Red::Prod, None) => (Some($S(a.prod())), None),
                        (Red::Min, None) => (Some($S(a.min())), None),
                        (Red::Max, None) => (Some($S(a.max())), None),
                        (Red::ArgMin, None) => (Some(Scalar::Int(a.argmin() as i64)), None),
                        (Red::ArgMax, None) => (Some(Scalar::Int(a.argmax() as i64)), None),
                        (Red::Sum, Some(ax)) => (None, Some($D(a.sum_axis(ax)))),
                        (Red::Prod, Some(ax)) => (None, Some($D(a.prod_axis(ax)))),
                        (Red::Min, Some(ax)) => (None, Some($D(a.min_axis(ax)))),
                        (Red::Max, Some(ax)) => (None, Some($D(a.max_axis(ax)))),
                        (Red::ArgMin, Some(ax)) => {
                            (None, Some(Data::I64(a.argmin_axis(ax).astype::<i64>())))
                        }
                        (Red::ArgMax, Some(ax)) => {
                            (None, Some(Data::I64(a.argmax_axis(ax).astype::<i64>())))
                        }
                        _ => unreachable!(),
                    }
                }};
            }
            macro_rules! red_float {
                ($a:expr, $S:path, $D:path) => {{
                    let a = $a;
                    match (red, axis) {
                        (Red::Mean, None) => (Some(Scalar::Float(f64::from(a.mean()))), None),
                        (Red::Std, None) => (Some(Scalar::Float(f64::from(a.std()))), None),
                        (Red::Var, None) => (Some(Scalar::Float(f64::from(a.var()))), None),
                        (Red::Mean, Some(ax)) => (None, Some($D(a.mean_axis(ax)))),
                        (Red::Std, Some(ax)) => (None, Some($D(a.std_axis(ax)))),
                        (Red::Var, Some(ax)) => (None, Some($D(a.var_axis(ax)))),
                        _ => red_num!(a, $S, $D),
                    }
                }};
            }
            Ok(match &d {
                Data::F64(a) => red_float!(a, Scalar::Float, Data::F64),
                Data::F32(a) => {
                    let (s, arr) = red_float!(a, scalar_f32, Data::F32);
                    (s, arr)
                }
                Data::I64(a) => red_num!(a, Scalar::Int, Data::I64),
                Data::Bool(a) => match (red, axis) {
                    (Red::Min, None) => (Some(Scalar::Bool(a.all())), None),
                    (Red::Max, None) => (Some(Scalar::Bool(a.any())), None),
                    (Red::ArgMin, None) => {
                        (Some(Scalar::Int(a.astype::<i64>().argmin() as i64)), None)
                    }
                    (Red::ArgMax, None) => {
                        (Some(Scalar::Int(a.astype::<i64>().argmax() as i64)), None)
                    }
                    (Red::Min, Some(ax)) => (None, Some(Data::Bool(a.all_axis(ax)))),
                    (Red::Max, Some(ax)) => (None, Some(Data::Bool(a.any_axis(ax)))),
                    (Red::ArgMin, Some(ax)) => (
                        None,
                        Some(Data::I64(a.astype::<i64>().argmin_axis(ax).astype::<i64>())),
                    ),
                    (Red::ArgMax, Some(ax)) => (
                        None,
                        Some(Data::I64(a.astype::<i64>().argmax_axis(ax).astype::<i64>())),
                    ),
                    _ => unreachable!(),
                },
            })
        })??;
        reduction_result(py, whole, arr)
    }
}

fn bool_op(
    a: &PyArray,
    o: &Bound<'_, PyAny>,
    f: impl Fn(&Array<bool>, &Array<bool>) -> Array<bool>,
) -> PyResult<PyArray> {
    let right = match operand(o)? {
        Operand::Array(d) => d,
        Operand::Scalar(s) => scalar_data(s, DType::Bool),
    };
    match (&a.data, right.cast(DType::Bool)) {
        (Data::Bool(x), Data::Bool(y)) => Ok(PyArray::new(Data::Bool(guard(|| f(x, &y))?))),
        _ => Err(PyTypeError::new_err(
            "bitwise operations require boolean arrays",
        )),
    }
}

fn nested(
    py: Python<'_>,
    flat: &[Scalar],
    shape: &[usize],
    pos: &mut usize,
) -> PyResult<Py<PyAny>> {
    if shape.is_empty() {
        let s = flat[*pos];
        *pos += 1;
        return scalar_to_py(py, s);
    }
    let list = PyList::empty(py);
    for _ in 0..shape[0] {
        list.append(nested(py, flat, &shape[1..], pos)?)?;
    }
    Ok(list.into_any().unbind())
}

/// Translates a Python index (int, slice, None, Ellipsis, list/array of ints, bool mask, tuple of
/// those) into rustnum index arguments. The flag is true when only plain integers were used.
fn parse_key(shape: &[usize], key: &Bound<'_, PyAny>) -> PyResult<(Vec<SliceArg>, bool)> {
    let items: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
        Ok(t) => t.iter().collect(),
        Err(_) => vec![key.clone()],
    };
    let consumed = items
        .iter()
        .filter(|i| !i.is_none() && !i.is(key.py().Ellipsis()))
        .count();
    let mut args = Vec::new();
    let mut axis = 0usize;
    let mut all_int = true;
    for item in &items {
        if item.is(key.py().Ellipsis()) {
            all_int = false;
            let fill = shape.len().saturating_sub(consumed);
            for _ in 0..fill {
                args.push(SliceArg::all());
                axis += 1;
            }
        } else if item.is_none() {
            all_int = false;
            args.push(SliceArg::NewAxis);
        } else if let Ok(s) = item.cast::<PySlice>() {
            all_int = false;
            let get = |name: &str| -> PyResult<Option<isize>> {
                let v = s.getattr(name)?;
                if v.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(v.extract::<isize>()?))
                }
            };
            let step = get("step")?.unwrap_or(1);
            if step == 0 {
                return Err(PyValueError::new_err("slice step cannot be zero"));
            }
            args.push(SliceArg::Range {
                start: get("start")?,
                end: get("stop")?,
                step,
            });
            axis += 1;
        } else if let Some(sc) = Scalar::from_py(item) {
            let i = match sc {
                Scalar::Int(i) => i as isize,
                Scalar::Bool(_) | Scalar::Float(_) => {
                    return Err(PyIndexError::new_err(
                        "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices",
                    ));
                }
            };
            let n = *shape.get(axis).ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "too many indices for array: array is {}-dimensional, but {} were indexed",
                    shape.len(),
                    consumed
                ))
            })? as isize;
            if i < -n || i >= n {
                return Err(PyIndexError::new_err(format!(
                    "index {i} is out of bounds for axis {axis} with size {n}"
                )));
            }
            args.push(SliceArg::Index(i));
            axis += 1;
        } else {
            all_int = false;
            // list or array of integers / booleans
            let (idx, is_bool) = match operand(item)? {
                Operand::Array(Data::Bool(m)) => (None, Some(m)),
                Operand::Array(Data::I64(a)) => (Some(a), None),
                Operand::Array(_) => {
                    return Err(PyIndexError::new_err(
                        "arrays used as indices must be of integer (or boolean) type",
                    ));
                }
                Operand::Scalar(_) => unreachable!(),
            };
            if let Some(m) = is_bool {
                args.push(guard(|| SliceArg::from(&m))?);
            } else if let Some(a) = idx {
                let n = *shape
                    .get(axis)
                    .ok_or_else(|| PyIndexError::new_err("too many indices for array"))?
                    as i64;
                if let Some(bad) = a.iter().find(|&v| v < -n || v >= n) {
                    return Err(PyIndexError::new_err(format!(
                        "index {bad} is out of bounds for axis {axis} with size {n}"
                    )));
                }
                args.push(SliceArg::from(&a));
            }
            axis += 1;
        }
    }
    Ok((args, all_int))
}

// ---------------------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------------------

fn matmul(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let t = promote(a.data.dtype(), b.data.dtype()).max(DType::I64);
    let (l, r) = (a.data.cast(t), b.data.cast(t));
    let out = guard(|| match (&l, &r) {
        (Data::F64(x), Data::F64(y)) => Data::F64(x.matmul(y)),
        (Data::F32(x), Data::F32(y)) => Data::F32(x.matmul(y)),
        (Data::I64(x), Data::I64(y)) => Data::I64(x.matmul(y)),
        _ => unreachable!(),
    })?;
    Ok(PyArray::new(out))
}

fn dot(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let t = promote(a.data.dtype(), b.data.dtype()).max(DType::I64);
    let (l, r) = (a.data.cast(t), b.data.cast(t));
    let out = guard(|| match (&l, &r) {
        (Data::F64(x), Data::F64(y)) => Data::F64(x.dot(y)),
        (Data::F32(x), Data::F32(y)) => Data::F32(x.dot(y)),
        (Data::I64(x), Data::I64(y)) => Data::I64(x.dot(y)),
        _ => unreachable!(),
    })?;
    Ok(PyArray::new(out))
}

fn to_array(o: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    match operand(o)? {
        Operand::Array(d) => Ok(PyArray::new(d)),
        Operand::Scalar(s) => Ok(PyArray::new(scalar_data(s, s.dtype()))),
    }
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
fn array(obj: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let want = parse_dtype(dtype)?;
    if let Ok(a) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(PyArray::new(match want {
            Some(t) => a.data.cast(t),
            None => each_same!(&a.data, x => x.copy()),
        }));
    }
    // A NumPy array keeps its dtype when it is one of ours (float32 stays float32).
    let want = want.or_else(|| {
        let name = obj.getattr("dtype").ok()?.getattr("name").ok()?;
        DType::parse(&name.extract::<String>().ok()?).ok()
    });
    let mut shape = vec![];
    let mut out = vec![];
    flatten(obj, &mut shape, 0, &mut out)?;
    Ok(PyArray::new(build(&out, &shape, want)?))
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
fn asarray(obj: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    array(obj, dtype)
}

fn filled(shape: &[usize], v: f64, dtype: Option<DType>) -> Data {
    match dtype.unwrap_or(DType::F64) {
        DType::Bool => Data::Bool(np::full(shape, v != 0.0)),
        DType::I64 => Data::I64(np::full(shape, v as i64)),
        DType::F32 => Data::F32(np::full(shape, v as f32)),
        DType::F64 => Data::F64(np::full(shape, v)),
    }
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn zeros(shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    Ok(PyArray::new(filled(
        &parse_shape(shape)?,
        0.0,
        parse_dtype(dtype)?,
    )))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn empty(shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    zeros(shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn ones(shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    Ok(PyArray::new(filled(
        &parse_shape(shape)?,
        1.0,
        parse_dtype(dtype)?,
    )))
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None))]
fn full(
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let s = Scalar::from_py(fill_value)
        .ok_or_else(|| PyTypeError::new_err("fill_value must be a number"))?;
    let t = parse_dtype(dtype)?.unwrap_or(s.dtype());
    Ok(PyArray::new(filled(
        &parse_shape(shape)?,
        s.as_f64(),
        Some(t),
    )))
}

#[pyfunction]
#[pyo3(signature = (n, m=None, dtype=None))]
fn eye(n: usize, m: Option<usize>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let t = parse_dtype(dtype)?.unwrap_or(DType::F64);
    let m = m.unwrap_or(n);
    let mut d = filled(&[n, m], 0.0, Some(t));
    for i in 0..n.min(m) {
        match &mut d {
            Data::F64(a) => a.set(&[i, i], 1.0),
            Data::F32(a) => a.set(&[i, i], 1.0),
            Data::I64(a) => a.set(&[i, i], 1),
            Data::Bool(a) => a.set(&[i, i], true),
        }
    }
    Ok(PyArray::new(d))
}

#[pyfunction]
#[pyo3(signature = (start, stop=None, step=None, dtype=None))]
fn arange(
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let sc = |o: &Bound<'_, PyAny>| {
        Scalar::from_py(o).ok_or_else(|| PyTypeError::new_err("arange arguments must be numbers"))
    };
    let (lo, hi) = match stop {
        Some(s) => (sc(start)?, sc(s)?),
        None => (Scalar::Int(0), sc(start)?),
    };
    let st = match step {
        Some(s) => sc(s)?,
        None => Scalar::Int(1),
    };
    let float = [lo, hi, st].iter().any(|s| matches!(s, Scalar::Float(_)));
    let t = parse_dtype(dtype)?.unwrap_or(if float { DType::F64 } else { DType::I64 });
    if st.as_f64() == 0.0 {
        return Err(PyValueError::new_err("Maximum allowed size exceeded"));
    }
    let out = guard(|| match t {
        DType::F64 => Data::F64(np::arange(lo.as_f64(), hi.as_f64(), st.as_f64())),
        DType::F32 => Data::F32(np::arange(
            lo.as_f64() as f32,
            hi.as_f64() as f32,
            st.as_f64() as f32,
        )),
        DType::I64 => Data::I64(np::arange(lo.as_i64(), hi.as_i64(), st.as_i64())),
        DType::Bool => {
            Data::I64(np::arange(lo.as_i64(), hi.as_i64(), st.as_i64())).cast(DType::Bool)
        }
    })?;
    Ok(PyArray::new(out))
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50))]
fn linspace(start: f64, stop: f64, num: usize) -> PyArray {
    PyArray::new(Data::F64(np::linspace(start, stop, num)))
}

#[pyfunction]
fn matmul_py(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    matmul(&to_array(a)?, &to_array(b)?)
}

#[pyfunction(name = "dot")]
fn dot_py(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    dot(&to_array(a)?, &to_array(b)?)
}

macro_rules! unary_fn {
    ($($name:ident => $m:ident),*) => {$(
        #[pyfunction]
        fn $name(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
            let a = to_array(a)?;
            let d = if a.data.dtype() < DType::F32 { a.data.cast(DType::F64) } else { a.data.clone() };
            float_ops_only(&d)?;
            Ok(PyArray::new(match &d {
                Data::F64(x) => Data::F64(x.$m()),
                Data::F32(x) => Data::F32(x.$m()),
                _ => unreachable!(),
            }))
        }
    )*};
}
unary_fn!(exp => exp, log => log, sqrt => sqrt, sin => sin, cos => cos, tan => tan, tanh => tanh, floor => floor, ceil => ceil);

#[pyfunction(name = "abs")]
fn abs_py(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    to_array(a)?.__abs__()
}

#[pyfunction]
fn maximum(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let l = to_array(a)?;
    Ok(PyArray::new(binary(Op::Max, &l.data, operand(b)?, false)?))
}

#[pyfunction]
fn minimum(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let l = to_array(a)?;
    Ok(PyArray::new(binary(Op::Min, &l.data, operand(b)?, false)?))
}

#[pyfunction(name = "where")]
fn where_py(
    cond: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<PyArray> {
    let c = to_array(cond)?.data.cast(DType::Bool);
    let (xo, yo) = (operand(x)?, operand(y)?);
    let dt = |o: &Operand| match o {
        Operand::Array(d) => d.dtype(),
        Operand::Scalar(s) => s.dtype(),
    };
    let t = promote(dt(&xo), dt(&yo));
    let cast = |o: Operand| match o {
        Operand::Array(d) => d.cast(t),
        Operand::Scalar(s) => scalar_data(s, t),
    };
    let (xd, yd) = (cast(xo), cast(yo));
    let Data::Bool(c) = c else { unreachable!() };
    let out = guard(|| match (&xd, &yd) {
        (Data::F64(a), Data::F64(b)) => Data::F64(np::where_(&c, a, b)),
        (Data::F32(a), Data::F32(b)) => Data::F32(np::where_(&c, a, b)),
        (Data::I64(a), Data::I64(b)) => Data::I64(np::where_(&c, a, b)),
        (Data::Bool(a), Data::Bool(b)) => Data::Bool(np::where_(&c, a, b)),
        _ => unreachable!(),
    })?;
    Ok(PyArray::new(out))
}

fn gather(items: &Bound<'_, PyAny>) -> PyResult<(Vec<Data>, DType)> {
    let list: Vec<Bound<'_, PyAny>> = items.try_iter()?.collect::<PyResult<_>>()?;
    let arrays: Vec<PyArray> = list.iter().map(|o| to_array(o)).collect::<PyResult<_>>()?;
    let t = arrays
        .iter()
        .map(|a| a.data.dtype())
        .reduce(promote)
        .ok_or_else(|| PyValueError::new_err("need at least one array to concatenate"))?;
    Ok((arrays.iter().map(|a| a.data.cast(t)).collect(), t))
}

macro_rules! stack_like {
    ($name:ident, $f:path) => {
        #[pyfunction]
        #[pyo3(signature = (arrays, axis=0))]
        fn $name(arrays: &Bound<'_, PyAny>, axis: isize) -> PyResult<PyArray> {
            let (ds, t) = gather(arrays)?;
            macro_rules! go {
                ($V:path) => {{
                    let refs: Vec<_> = ds
                        .iter()
                        .map(|d| match d {
                            $V(a) => a,
                            _ => unreachable!(),
                        })
                        .collect();
                    $V(guard(|| $f(&refs, axis))?)
                }};
            }
            Ok(PyArray::new(match t {
                DType::F64 => go!(Data::F64),
                DType::F32 => go!(Data::F32),
                DType::I64 => go!(Data::I64),
                DType::Bool => go!(Data::Bool),
            }))
        }
    };
}
stack_like!(concatenate, np::concatenate);
stack_like!(stack, np::stack);

#[pyfunction]
fn transpose(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    Ok(to_array(a)?.T())
}

macro_rules! reduce_fn {
    ($($name:ident),*) => {$(
        #[pyfunction]
        #[pyo3(signature = (a, axis=None))]
        fn $name(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
            to_array(a)?.$name(py, axis)
        }
    )*};
}
reduce_fn!(sum, prod, mean, var, min, max, argmin, argmax);

#[pyfunction(name = "std")]
#[pyo3(signature = (a, axis=None))]
fn std_py(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    to_array(a)?.std(py, axis)
}

// ---------------------------------------------------------------------------------------
// linalg
// ---------------------------------------------------------------------------------------

fn float_pair(a: &PyArray) -> Data {
    if a.data.dtype() < DType::F32 {
        a.data.cast(DType::F64)
    } else {
        a.data.clone()
    }
}

macro_rules! linalg_unary {
    ($name:ident, $f:path) => {
        #[pyfunction]
        fn $name(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
            let d = float_pair(&to_array(a)?);
            let out = guard(|| match &d {
                Data::F64(x) => $f(x).map(Data::F64).map_err(np_err),
                Data::F32(x) => $f(x).map(Data::F32).map_err(np_err),
                _ => unreachable!(),
            })??;
            Ok(PyArray::new(out))
        }
    };
}
linalg_unary!(linalg_inv, np::linalg::inv);
linalg_unary!(linalg_cholesky, np::linalg::cholesky);

#[pyfunction]
fn linalg_solve(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let (a, b) = (to_array(a)?, to_array(b)?);
    let t = promote(a.data.dtype(), b.data.dtype()).max(DType::F32);
    let (l, r) = (a.data.cast(t), b.data.cast(t));
    let out = guard(|| match (&l, &r) {
        (Data::F64(x), Data::F64(y)) => np::linalg::solve(x, y).map(Data::F64).map_err(np_err),
        (Data::F32(x), Data::F32(y)) => np::linalg::solve(x, y).map(Data::F32).map_err(np_err),
        _ => unreachable!(),
    })??;
    Ok(PyArray::new(out))
}

#[pyfunction]
fn linalg_det(a: &Bound<'_, PyAny>) -> PyResult<f64> {
    let d = float_pair(&to_array(a)?);
    guard(|| match &d {
        Data::F64(x) => np::linalg::det(x),
        Data::F32(x) => f64::from(np::linalg::det(x)),
        _ => unreachable!(),
    })
}

#[pyfunction]
fn linalg_norm(a: &Bound<'_, PyAny>) -> PyResult<f64> {
    let d = float_pair(&to_array(a)?);
    guard(|| match &d {
        Data::F64(x) => np::linalg::norm(x),
        Data::F32(x) => f64::from(np::linalg::norm(x)),
        _ => unreachable!(),
    })
}

#[pyfunction]
fn linalg_pinv(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let d = float_pair(&to_array(a)?);
    let out = guard(|| match &d {
        Data::F64(x) => Data::F64(np::linalg::pinv(x)),
        Data::F32(x) => Data::F32(np::linalg::pinv(x)),
        _ => unreachable!(),
    })?;
    Ok(PyArray::new(out))
}

#[pyfunction]
fn linalg_svd(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let d = float_pair(&to_array(a)?);
    let (u, s, vt) = guard(|| match &d {
        Data::F64(x) => {
            let (u, s, v) = np::linalg::svd(x);
            (Data::F64(u), Data::F64(s), Data::F64(v))
        }
        Data::F32(x) => {
            let (u, s, v) = np::linalg::svd(x);
            (Data::F32(u), Data::F32(s), Data::F32(v))
        }
        _ => unreachable!(),
    })?;
    let t = PyTuple::new(py, [PyArray::new(u), PyArray::new(s), PyArray::new(vt)])?;
    Ok(t.into_any().unbind())
}

#[pyfunction]
fn linalg_eigh(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let d = float_pair(&to_array(a)?);
    let (w, v) = guard(|| match &d {
        Data::F64(x) => {
            let (w, v) = np::linalg::eigh(x);
            (Data::F64(w), Data::F64(v))
        }
        Data::F32(x) => {
            let (w, v) = np::linalg::eigh(x);
            (Data::F32(w), Data::F32(v))
        }
        _ => unreachable!(),
    })?;
    Ok(PyTuple::new(py, [PyArray::new(w), PyArray::new(v)])?
        .into_any()
        .unbind())
}

#[pyfunction]
fn linalg_qr(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let d = float_pair(&to_array(a)?);
    let (q, r) = guard(|| match &d {
        Data::F64(x) => {
            let (q, r) = np::linalg::qr(x);
            (Data::F64(q), Data::F64(r))
        }
        Data::F32(x) => {
            let (q, r) = np::linalg::qr(x);
            (Data::F32(q), Data::F32(r))
        }
        _ => unreachable!(),
    })?;
    Ok(PyTuple::new(py, [PyArray::new(q), PyArray::new(r)])?
        .into_any()
        .unbind())
}

// ---------------------------------------------------------------------------------------
// random
// ---------------------------------------------------------------------------------------

static GLOBAL_RNG: Mutex<Option<np::random::Rng>> = Mutex::new(None);

fn with_rng<R>(f: impl FnOnce(&mut np::random::Rng) -> R) -> R {
    let mut g = GLOBAL_RNG.lock().unwrap_or_else(|e| e.into_inner());
    let rng = g.get_or_insert_with(np::random::Rng::from_entropy);
    f(rng)
}

fn shape_args(args: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    if args.len() == 1 {
        let first = args.get_item(0)?;
        if let Ok(v) = first.extract::<Vec<usize>>() {
            return Ok(v);
        }
    }
    args.extract()
}

#[pyfunction]
fn random_seed(seed: u64) {
    *GLOBAL_RNG.lock().unwrap_or_else(|e| e.into_inner()) = Some(np::random::Rng::seed(seed));
}

#[pyfunction]
#[pyo3(signature = (*shape))]
fn random_rand(shape: &Bound<'_, PyTuple>) -> PyResult<PyArray> {
    let s = shape_args(shape)?;
    Ok(PyArray::new(Data::F64(with_rng(|r| r.random(&s)))))
}

#[pyfunction]
#[pyo3(signature = (*shape))]
fn random_randn(shape: &Bound<'_, PyTuple>) -> PyResult<PyArray> {
    let s = shape_args(shape)?;
    Ok(PyArray::new(Data::F64(with_rng(|r| r.standard_normal(&s)))))
}

#[pyfunction]
#[pyo3(signature = (low=0.0, high=1.0, size=None))]
fn random_uniform(low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let s = size.map(parse_shape).transpose()?.unwrap_or_default();
    Ok(PyArray::new(Data::F64(with_rng(|r| {
        r.uniform(low, high, &s)
    }))))
}

#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
fn random_normal(loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let s = size.map(parse_shape).transpose()?.unwrap_or_default();
    Ok(PyArray::new(Data::F64(with_rng(|r| {
        r.normal(loc, scale, &s)
    }))))
}

#[pyfunction]
#[pyo3(signature = (low, high=None, size=None))]
fn random_randint(
    low: i64,
    high: Option<i64>,
    size: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let (lo, hi) = match high {
        Some(h) => (low, h),
        None => (0, low),
    };
    let s = size.map(parse_shape).transpose()?.unwrap_or_default();
    let out = guard(|| with_rng(|r| r.integers(lo, hi, &s)))?;
    Ok(PyArray::new(Data::I64(out)))
}

// ---------------------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------------------

/// Registers `name` on the submodule, under its Python-facing name.
fn add_fn(
    m: &Bound<'_, PyModule>,
    f: Bound<'_, pyo3::types::PyCFunction>,
    name: &str,
) -> PyResult<()> {
    m.add(name, f)
}

#[pymodule]
fn rustnum_py(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add("float64", "float64")?;
    m.add("float32", "float32")?;
    m.add("int64", "int64")?;
    m.add("bool_", "bool")?;
    m.add("pi", std::f64::consts::PI)?;
    m.add("e", std::f64::consts::E)?;
    m.add("nan", f64::NAN)?;
    m.add("inf", f64::INFINITY)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    add_fn(m, wrap_pyfunction!(matmul_py, m)?, "matmul")?;
    add_fn(m, wrap_pyfunction!(dot_py, m)?, "dot")?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    add_fn(m, wrap_pyfunction!(abs_py, m)?, "abs")?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    add_fn(m, wrap_pyfunction!(where_py, m)?, "where")?;
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(std_py, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;

    let linalg = PyModule::new(py, "linalg")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_inv, &linalg)?, "inv")?;
    add_fn(
        &linalg,
        wrap_pyfunction!(linalg_cholesky, &linalg)?,
        "cholesky",
    )?;
    add_fn(&linalg, wrap_pyfunction!(linalg_solve, &linalg)?, "solve")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_det, &linalg)?, "det")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_norm, &linalg)?, "norm")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_pinv, &linalg)?, "pinv")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_svd, &linalg)?, "svd")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_eigh, &linalg)?, "eigh")?;
    add_fn(&linalg, wrap_pyfunction!(linalg_qr, &linalg)?, "qr")?;
    m.add_submodule(&linalg)?;

    let random = PyModule::new(py, "random")?;
    add_fn(&random, wrap_pyfunction!(random_seed, &random)?, "seed")?;
    add_fn(&random, wrap_pyfunction!(random_rand, &random)?, "rand")?;
    add_fn(&random, wrap_pyfunction!(random_randn, &random)?, "randn")?;
    add_fn(
        &random,
        wrap_pyfunction!(random_uniform, &random)?,
        "uniform",
    )?;
    add_fn(&random, wrap_pyfunction!(random_normal, &random)?, "normal")?;
    add_fn(
        &random,
        wrap_pyfunction!(random_randint, &random)?,
        "randint",
    )?;
    m.add_submodule(&random)?;
    // make `import rustnum_py.linalg` work
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("rustnum_py.linalg", &linalg)?;
    sys_modules.set_item("rustnum_py.random", &random)?;
    Ok(())
}
