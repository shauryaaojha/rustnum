//! datetime64 / timedelta64: values here are hand-checked against NumPy 2.3 output; the
//! bulk comparison (every unit, arithmetic, casts, parsing) lives in the parity suite.

use rustnum as np;
use rustnum::datetime::{D, M, Y, h, m, ms, ns, s};
use rustnum::{Array, DateTime64, TimeDelta64};
use std::process::Command;

type Day = DateTime64<D>;
type Days = TimeDelta64<D>;

fn days(texts: &[&str]) -> Array<Day> {
    Array::from(
        texts
            .iter()
            .map(|t| Day::parse(t).unwrap())
            .collect::<Vec<_>>(),
    )
}

#[test]
fn numpy_array_text() {
    let a = days(&["2024-01-01", "NaT", "2024-03-05"]);
    assert_eq!(a.to_string(), "['2024-01-01'        'NaT' '2024-03-05']");
    assert_eq!(
        format!("{a:?}"),
        "array(['2024-01-01',        'NaT', '2024-03-05'], dtype='datetime64[D]')"
    );
    let t = Array::from(vec![Days::new(1), Days::new(-2), Days::NAT]);
    assert_eq!(t.to_string(), "[    1    -2 'NaT']");
    assert_eq!(
        format!("{t:?}"),
        "array([    1,    -2, 'NaT'], dtype='timedelta64[D]')"
    );
    let empty: Array<Day> = Array::from(Vec::<Day>::new());
    assert_eq!(format!("{empty:?}"), "array([], dtype='datetime64[D]')");
}

#[test]
fn zero_dim_text() {
    let a = Array::scalar(Day::parse("2024-01-01").unwrap());
    assert_eq!(a.to_string(), "2024-01-01");
    assert_eq!(
        format!("{a:?}"),
        "array('2024-01-01', dtype='datetime64[D]')"
    );
    let t = Array::scalar(Days::new(5));
    assert_eq!(t.to_string(), "5 days");
    assert_eq!(format!("{t:?}"), "array(5, dtype='timedelta64[D]')");
    assert_eq!(Array::scalar(Day::NAT).to_string(), "NaT");
}

#[test]
fn scalar_text() {
    assert_eq!(Day::parse("2024-02-29").unwrap().to_string(), "2024-02-29");
    assert_eq!(
        format!("{:?}", Day::parse("2024-02-29").unwrap()),
        "numpy.datetime64('2024-02-29')"
    );
    assert_eq!(Days::new(1).to_string(), "1 days");
    assert_eq!(
        format!("{:?}", TimeDelta64::<s>::new(3)),
        "numpy.timedelta64(3,'s')"
    );
    assert_eq!(Days::NAT.to_string(), "NaT");
}

#[test]
fn parse_rules() {
    assert_eq!(Day::parse("1970-01-01").unwrap().value(), 0);
    assert_eq!(Day::parse("2024-02-29").unwrap().value(), 19782);
    assert_eq!(Day::parse("1969-12-31").unwrap().value(), -1);
    // 1900 is not a leap year, 2000 is.
    assert!(Day::parse("1900-02-29").is_err());
    assert_eq!(Day::parse("2000-02-29").unwrap().value(), 11016);
    assert!(Day::parse("2024-13-01").is_err());
    assert!(Day::parse("not a date").is_err());
    // finer than the unit is an error, coarser widens
    assert!(Day::parse("2024-01-01T10:00").is_err());
    assert_eq!(Day::parse("2024-03").unwrap().value(), 19783);
    assert_eq!(
        DateTime64::<s>::parse("1970-01-02").unwrap().value(),
        86_400
    );
    assert_eq!(
        DateTime64::<ns>::parse("1970-01-01T00:00:00.5")
            .unwrap()
            .value(),
        500_000_000
    );
    assert!(Day::parse("NaT").unwrap().is_nat());
    assert_eq!("2024-01-01".parse::<Day>().unwrap().value(), 19723);
    assert_eq!(DateTime64::<Y>::parse("2024").unwrap().value(), 54);
    assert_eq!(DateTime64::<M>::parse("1969-12").unwrap().value(), -1);
}

#[test]
fn scalar_arithmetic() {
    let a = Day::parse("2024-03-05").unwrap();
    let b = Day::parse("2024-01-01").unwrap();
    assert_eq!(a - b, Days::new(64));
    assert_eq!(b + Days::new(64), a);
    assert_eq!(Days::new(64) + b, a);
    assert_eq!(a - Days::new(64), b);
    assert_eq!(Days::new(3) * 4, Days::new(12));
    assert_eq!(4 * Days::new(3), Days::new(12));
    assert_eq!(Days::new(7) / 2, Days::new(3));
    assert_eq!(Days::new(-7) / 2, Days::new(-3));
    assert_eq!(Days::new(6) / Days::new(4), 1.5);
    assert_eq!(Days::new(-7).floor_div(Days::new(2)), -4);
    assert_eq!(Days::new(-7) % Days::new(4), Days::new(1));
    assert_eq!(-Days::new(2), Days::new(-2));
    // NaT propagates and never compares equal
    assert!((a - Day::NAT).is_nat());
    assert!(Day::NAT != Day::NAT);
    assert!(Day::NAT.partial_cmp(&a).is_none());
    assert!((Days::NAT * 2).is_nat());
    assert!((Days::new(1) / Days::new(0)).is_infinite());
    assert!((Days::new(1) % Days::new(0)).is_nat());
}

#[test]
fn unit_conversion() {
    let d = Day::parse("2024-03-05").unwrap();
    assert_eq!(d.cast::<s>().to_string(), "2024-03-05T00:00:00");
    assert_eq!(d.cast::<ms>().value(), 19787 * 86_400_000);
    assert_eq!(d.cast::<M>().to_string(), "2024-03");
    assert_eq!(d.cast::<Y>().to_string(), "2024");
    // coarser targets floor toward minus infinity
    let before = Day::parse("1969-12-31").unwrap();
    assert_eq!(before.cast::<M>().value(), -1);
    assert_eq!(before.cast::<Y>().value(), -1);
    assert_eq!(DateTime64::<s>::new(-1).cast::<D>().value(), -1);
    // months to days use the calendar, time deltas use the average year / month
    assert_eq!(
        DateTime64::<M>::new(14).cast::<D>().to_string(),
        "1971-03-01"
    );
    assert_eq!(TimeDelta64::<Y>::new(10).cast::<D>().value(), 3652);
    assert_eq!(TimeDelta64::<M>::new(1).cast::<D>().value(), 30);
    assert_eq!(TimeDelta64::<h>::new(90).cast::<m>().value(), 5400);
    // NaT survives
    assert!(DateTime64::<D>::NAT.cast::<ns>().is_nat());
    assert!(TimeDelta64::<D>::NAT.cast::<s>().is_nat());
}

#[test]
fn array_ops_and_astype() {
    let a = days(&["2024-01-01", "2024-02-01", "2024-03-01"]);
    let b = days(&["2023-12-25", "2024-02-01", "2024-03-09"]);
    let d = &a - &b;
    assert_eq!(d.iter().map(|t| t.value()).collect::<Vec<_>>(), [7, 0, -8]);
    assert_eq!(
        (&b + &d).to_vec(),
        a.to_vec(),
        "b + (a - b) must give back a"
    );
    assert_eq!(a.lt(&b).to_vec(), [false, false, true]);
    assert_eq!(a.eq_elem(&b).to_vec(), [false, true, false]);
    let secs = a.astype::<DateTime64<s>>();
    assert_eq!(secs.get(&[1]).value(), 19754 * 86_400);
    assert_eq!(secs.astype::<Day>().to_vec(), a.to_vec());
    // number casts keep the raw count; NaN becomes NaT
    let x = np::array![0.0, 1.9, f64::NAN].astype::<Days>();
    assert_eq!(x.get(&[1]).value(), 1);
    assert!(x.get(&[2]).is_nat());
    let raw = a.astype::<i64>();
    assert_eq!(raw.to_vec(), [19723, 19754, 19783]);
    assert_eq!(raw.astype::<Day>().to_vec(), a.to_vec());
    // nanosecond counts survive an i64 round trip without a float detour
    let big = np::array![9_007_199_254_740_993_i64];
    assert_eq!(
        big.astype::<DateTime64<ns>>().astype::<i64>().to_vec(),
        [9_007_199_254_740_993]
    );
    let t = Array::from(vec![Days::new(1), Days::new(2), Days::new(3)]);
    assert_eq!(t.sum(), Days::new(6));
    assert_eq!(t.mean(), Days::new(2));
    assert_eq!(
        (&t * 2).to_vec(),
        [Days::new(2), Days::new(4), Days::new(6)]
    );
    assert_eq!(
        t.cumsum().to_vec(),
        [Days::new(1), Days::new(3), Days::new(6)]
    );
    assert_eq!(t.floor_div(Days::new(2)).to_vec(), [0, 1, 1]);
    let r = &t / &Array::from(vec![Days::new(2); 3]);
    assert_eq!(r.to_vec(), [0.5, 1.0, 1.5]);
}

#[test]
fn nat_sorts_last() {
    let a = days(&["2024-03-01", "NaT", "2024-01-01"]);
    assert_eq!(a.argsort().to_vec(), [2, 0, 1]);
    let sorted = a.sort();
    assert_eq!(sorted.get(&[0]).to_string(), "2024-01-01");
    assert!(sorted.get(&[2]).is_nat());
    assert_eq!(np::isnat(&a).to_vec(), [false, true, false]);
}

#[test]
fn arange_and_strings() {
    let a = Array::<Day>::arange(
        Day::parse("2024-01-01").unwrap(),
        Day::parse("2024-01-10").unwrap(),
        Days::new(4),
    );
    assert_eq!(
        np::datetime_as_string(&a),
        ["2024-01-01", "2024-01-05", "2024-01-09"]
    );
    let down = Array::<Day>::arange(
        Day::parse("2024-01-10").unwrap(),
        Day::parse("2024-01-01").unwrap(),
        Days::new(-4),
    );
    assert_eq!(down.size(), 3);
    let e = Array::<Day>::arange(Day::new(5), Day::new(0), Days::new(1));
    assert_eq!(e.size(), 0);
}

#[test]
fn npy_bytes_round_trip() {
    let a = days(&["2024-01-01", "NaT", "1969-12-31"]).reshape(&[1, 3]);
    let bytes = np::io::to_npy_bytes(&a);
    let head = String::from_utf8_lossy(&bytes[..80]).to_string();
    assert!(head.contains("'descr': '<M8[D]'"), "{head}");
    let b: Array<Day> = np::io::from_npy_bytes(&bytes).unwrap();
    assert_eq!(b.shape(), [1, 3]);
    assert!(b.get(&[0, 1]).is_nat());
    assert_eq!(b.get(&[0, 2]).value(), -1);
    // another unit or a timedelta of the same unit is a dtype error, never a silent cast
    assert!(matches!(
        np::io::from_npy_bytes::<DateTime64<s>>(&bytes),
        Err(np::Error::DType { .. })
    ));
    assert!(matches!(
        np::io::from_npy_bytes::<Days>(&bytes),
        Err(np::Error::DType { .. })
    ));
    let t = Array::from(vec![TimeDelta64::<ms>::new(5), TimeDelta64::NAT]);
    let tb = np::io::to_npy_bytes(&t);
    assert!(
        String::from_utf8_lossy(&tb[..80]).contains("'<m8[ms]'"),
        "{}",
        String::from_utf8_lossy(&tb[..80])
    );
    assert_eq!(
        np::io::from_npy_bytes::<TimeDelta64<ms>>(&tb)
            .unwrap()
            .get(&[0])
            .value(),
        5
    );
}

#[test]
fn npz_round_trip() {
    let dir = std::env::temp_dir().join(format!("rustnum_dt_npz_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("d.npz");
    let a = days(&["2000-02-29", "NaT"]);
    np::io::savez(&path, &[("when", &a)]).unwrap();
    let back = np::io::load_npz::<Day>(&path).unwrap();
    assert_eq!(back[0].0, "when");
    assert_eq!(back[0].1.get(&[0]).value(), 11016);
    assert!(back[0].1.get(&[1]).is_nat());
}

fn python_ok() -> bool {
    Command::new("python")
        .args(["-c", "import numpy"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn numpy_reads_what_rustnum_writes() {
    if !python_ok() {
        eprintln!("SKIP: python/numpy not available");
        return;
    }
    let dir = std::env::temp_dir().join(format!("rustnum_dt_py_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let d = dir.join("d.npy");
    let t = dir.join("t.npy");
    np::io::save_npy(&d, &days(&["2024-02-29", "NaT", "1969-12-31"])).unwrap();
    np::io::save_npy(
        &t,
        &Array::from(vec![
            TimeDelta64::<ns>::new(-5),
            TimeDelta64::NAT,
            TimeDelta64::new(7),
        ]),
    )
    .unwrap();
    let script = format!(
        "import numpy as np\n\
         d=np.load(r'{}'); t=np.load(r'{}')\n\
         assert d.dtype==np.dtype('datetime64[D]'), d.dtype\n\
         assert str(d)==\"['2024-02-29'        'NaT' '1969-12-31']\", str(d)\n\
         assert t.dtype==np.dtype('timedelta64[ns]'), t.dtype\n\
         assert t[0]==np.timedelta64(-5,'ns') and np.isnat(t[1]) and t[2]==np.timedelta64(7,'ns')\n\
         np.save(r'{}', np.array(['1999-12-31T23:59:59.250','NaT'],dtype='M8[ms]'))\n",
        d.display(),
        t.display(),
        dir.join("back.npy").display()
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
    let back: Array<DateTime64<ms>> = np::io::load_npy(dir.join("back.npy")).unwrap();
    assert_eq!(back.get(&[0]).to_string(), "1999-12-31T23:59:59.250");
    assert!(back.get(&[1]).is_nat());
}
