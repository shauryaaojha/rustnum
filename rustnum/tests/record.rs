//! Structured dtypes (`record!`): values are hand-checked against NumPy 2.3 output.

use rustnum as np;
use rustnum::Array;
use rustnum::datetime::D;
use std::process::Command;

np::record! {
    /// A 2-d point.
    pub struct Point { x: f64, y: f64 }
}

np::record! {
    pub struct Row { id: i32, flag: bool, w: f32 }
}

np::record! {
    pub struct Event { when: np::DateTime64<D>, gap: np::TimeDelta64<D>, z: np::Complex<f64>, h: np::f16 }
}

np::record! {
    pub struct Inner { p: i32, q: f32 }
}

np::record! {
    pub struct Outer { a: i32, n: Inner }
}

fn points() -> Array<Point> {
    Array::from(vec![
        Point { x: 3.0, y: 1.0 },
        Point { x: 1.0, y: 2.0 },
        Point { x: 1.0, y: -1.0 },
    ])
}

#[test]
fn field_access() {
    let a = points();
    assert_eq!(a.field::<f64>("x").to_vec(), [3.0, 1.0, 1.0]);
    assert_eq!(a.field::<f64>("y").to_vec(), [1.0, 2.0, -1.0]);
    assert_eq!(<Point as np::Record>::FIELDS, ["x", "y"]);
}

#[test]
#[should_panic(expected = "no field of name z")]
fn unknown_field_panics() {
    let _ = points().field::<f64>("z");
}

#[test]
#[should_panic(expected = "different dtype")]
fn wrong_field_type_panics() {
    let _ = points().field::<i32>("x");
}

#[test]
fn set_field_assigns_and_broadcasts() {
    let mut a = points();
    a.set_field("y", 0.5);
    assert_eq!(a.field::<f64>("y").to_vec(), [0.5, 0.5, 0.5]);
    a.set_field("x", np::array![7.0, 8.0, 9.0]);
    assert_eq!(a.field::<f64>("x").to_vec(), [7.0, 8.0, 9.0]);
}

#[test]
fn sort_and_argsort_by_field_order() {
    let a = points();
    // np.argsort(a, order=['x','y'], kind='stable') -> [2, 1, 0]
    assert_eq!(a.argsort_order(&["x", "y"]).to_vec(), [2, 1, 0]);
    // unnamed fields still break ties, so order=['x'] is the same -> [2, 1, 0]
    assert_eq!(a.argsort_order(&["x"]).to_vec(), [2, 1, 0]);
    assert_eq!(a.argsort_order(&["y"]).to_vec(), [2, 0, 1]);
    let sorted = a.sort_order(&["x", "y"]);
    assert_eq!(sorted.field::<f64>("y").to_vec(), [-1.0, 2.0, 1.0]);
}

#[test]
fn display_matches_numpy() {
    // np.array([(3.,1.),(1.,2.),(1.,-1.)], dtype=[('x','f8'),('y','f8')])
    assert_eq!(points().to_string(), "[(3.,  1.) (1.,  2.) (1., -1.)]");
    let rows = Array::from(vec![
        Row {
            id: 1,
            flag: true,
            w: 0.5,
        },
        Row {
            id: 22,
            flag: false,
            w: 10.25,
        },
    ]);
    assert_eq!(rows.to_string(), "[( 1,  True,  0.5 ) (22, False, 10.25)]");
    assert_eq!(
        format!(
            "{:?}",
            Array::from(vec![Point { x: 0.0, y: 0.0 }, Point { x: 0.0, y: 0.0 }])
        ),
        "array([(0., 0.), (0., 0.)], dtype=[('x', '<f8'), ('y', '<f8')])"
    );
}

#[test]
fn descr_strings() {
    assert_eq!(
        <Point as np::Record>::descr(),
        "[('x', '<f8'), ('y', '<f8')]"
    );
    assert_eq!(
        <Row as np::Record>::descr(),
        "[('id', '<i4'), ('flag', '|b1'), ('w', '<f4')]"
    );
    assert_eq!(
        <Outer as np::Record>::descr(),
        "[('a', '<i4'), ('n', [('p', '<i4'), ('q', '<f4')])]"
    );
    assert_eq!(
        <Event as np::Record>::descr(),
        "[('when', '<M8[D]'), ('gap', '<m8[D]'), ('z', '<c16'), ('h', '<f2')]"
    );
}

#[test]
fn npy_round_trip_in_rust() {
    let a = points();
    let bytes = np::io::to_npy_bytes(&a);
    let head = String::from_utf8_lossy(&bytes[..96]).to_string();
    assert!(
        head.contains("'descr': [('x', '<f8'), ('y', '<f8')]"),
        "{head}"
    );
    let b: Array<Point> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(b.to_vec(), a.to_vec());
    // a different record layout is a dtype error
    assert!(matches!(
        np::io::from_npy_bytes::<Row>(&bytes),
        Err(np::Error::DType { .. })
    ));
    // so is a plain float array
    assert!(np::io::from_npy_bytes::<f64>(&bytes).is_err());
}

#[test]
fn nested_and_exotic_fields_round_trip() {
    let o = Array::from(vec![
        Outer {
            a: 1,
            n: Inner { p: -2, q: 0.25 },
        },
        Outer {
            a: 3,
            n: Inner { p: 4, q: -8.5 },
        },
    ]);
    let back: Array<Outer> = np::io::from_npy_bytes(&np::io::to_npy_bytes(&o)).unwrap();
    assert_eq!(back.to_vec(), o.to_vec());
    let e = Array::from(vec![Event {
        when: np::DateTime64::<D>::parse("2024-02-29").unwrap(),
        gap: np::TimeDelta64::<D>::new(-3),
        z: np::Complex::new(1.0, -2.0),
        h: np::f16::from_f32(1.5),
    }]);
    let back: Array<Event> = np::io::from_npy_bytes(&np::io::to_npy_bytes(&e)).unwrap();
    assert_eq!(back.get(&[0]).when.value(), 19782);
    assert_eq!(back.get(&[0]).gap.value(), -3);
    assert_eq!(back.get(&[0]).z, np::Complex::new(1.0, -2.0));
    assert_eq!(back.get(&[0]).h.to_f32(), 1.5);
}

#[test]
fn npz_round_trip() {
    let dir = std::env::temp_dir().join(format!("rustnum_rec_npz_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("r.npz");
    np::io::savez(&path, &[("pts", &points())]).unwrap();
    let back = np::io::load_npz::<Point>(&path).unwrap();
    assert_eq!(back[0].1.to_vec(), points().to_vec());
}

fn python_ok() -> bool {
    Command::new("python")
        .args(["-c", "import numpy"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn numpy_reads_and_writes_the_same_files() {
    if !python_ok() {
        eprintln!("SKIP: python/numpy not available");
        return;
    }
    let dir = std::env::temp_dir().join(format!("rustnum_rec_py_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mine = dir.join("mine.npy");
    let theirs = dir.join("theirs.npy");
    np::io::save_npy(&mine, &points()).unwrap();
    let o = Array::from(vec![Outer {
        a: 5,
        n: Inner { p: 6, q: 7.5 },
    }]);
    np::io::save_npy(dir.join("outer.npy"), &o).unwrap();
    let script = format!(
        "import numpy as np\n\
         a=np.load(r'{mine}')\n\
         assert a.dtype==np.dtype([('x','<f8'),('y','<f8')]), a.dtype\n\
         assert a['x'].tolist()==[3.0,1.0,1.0] and a['y'].tolist()==[1.0,2.0,-1.0]\n\
         assert np.argsort(a,order=['x','y'],kind='stable').tolist()==[2,1,0]\n\
         o=np.load(r'{outer}')\n\
         assert o.dtype==np.dtype([('a','<i4'),('n',[('p','<i4'),('q','<f4')])]), o.dtype\n\
         assert o['a'][0]==5 and o['n']['p'][0]==6 and o['n']['q'][0]==7.5\n\
         b=np.array([(1.5,-2.0),(0.0,4.0)],dtype=[('x','<f8'),('y','<f8')])\n\
         np.save(r'{theirs}', b)\n",
        mine = mine.display(),
        outer = dir.join("outer.npy").display(),
        theirs = theirs.display()
    );
    let out = Command::new("python")
        .args(["-c", &script])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let back: Array<Point> = np::io::load_npy(&theirs).unwrap();
    assert_eq!(back.field::<f64>("x").to_vec(), [1.5, 0.0]);
    assert_eq!(back.field::<f64>("y").to_vec(), [-2.0, 4.0]);
}

#[test]
fn works_with_array_methods() {
    let a = points();
    assert_eq!(a.shape(), [3]);
    let r = a.reshape(&[3, 1]);
    assert_eq!(r.shape(), [3, 1]);
    let t = r.t();
    assert_eq!(t.shape(), [1, 3]);
    assert_eq!(a.get(&[1]).x, 1.0);
}
