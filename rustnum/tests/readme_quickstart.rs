//! Runs the README quick start so the README cannot go out of date.
//!
//! The body of `quick_start` is the same code as the first `rust` block in `README.md`
//! (with assertions added). `readme_contains_the_tested_lines` fails if the README text changes
//! and this test does not.

use rustnum as np;

#[test]
fn quick_start() {
    let a = np::arange(0.0, 6.0, 1.0).reshape(&[2, 3]); // np.arange(6.0).reshape(2, 3)
    let b = np::array![10.0, 20.0, 30.0]; // np.array([10.0, 20.0, 30.0])
    let c = &a + &b; // broadcasting: shape [2, 3] + shape [3]
    println!("{c}");
    println!("sum = {}", c.sum());
    println!("column means = {}", c.mean_axis(0)); // np.mean(c, axis=0)
    println!("second column = {}", c.slice(&np::s![.., 1])); // c[:, 1]
    println!("a @ a.T =\n{}", a.matmul(&a.t()));

    // Assertions on the values the README code produces.
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.to_vec(), vec![10.0, 21.0, 32.0, 13.0, 24.0, 35.0]);
    assert_eq!(c.sum(), 135.0);
    assert_eq!(c.mean_axis(0).to_vec(), vec![11.5, 22.5, 33.5]);
    assert_eq!(c.slice(&np::s![.., 1]).to_vec(), vec![21.0, 24.0]);
    assert_eq!(a.matmul(&a.t()).to_vec(), vec![5.0, 14.0, 14.0, 50.0]);
    assert_eq!(format!("{c}"), "[[10. 21. 32.]\n [13. 24. 35.]]");
}

#[test]
fn readme_contains_the_tested_lines() {
    let readme = include_str!("../../README.md");
    let lines = [
        "use rustnum as np;",
        "let a = np::arange(0.0, 6.0, 1.0).reshape(&[2, 3]);",
        "let b = np::array![10.0, 20.0, 30.0];",
        "let c = &a + &b;",
        "println!(\"{c}\");",
        "println!(\"sum = {}\", c.sum());",
        "println!(\"column means = {}\", c.mean_axis(0));",
        "println!(\"second column = {}\", c.slice(&np::s![.., 1]));",
        "println!(\"a @ a.T =\\n{}\", a.matmul(&a.t()));",
    ];
    for line in lines {
        assert!(
            readme.contains(line),
            "README.md no longer contains: {line}"
        );
    }
}
