//! interp1 at a duplicated lower knot takes the LAST of the plateau, as
//! np.interp does (#842). It returned the first, so a classic runner
//! priced at exactly zero mid-calibration read its offset off the wrong
//! end of the zero-price plateau and was sent across the field: a
//! self-generated five-runner book missed by 0.0152 where python's
//! reference calibration reaches 2.1e-6.
use winning::{exact_calibrate, exact_state_prices_from_offsets, interp1};

#[test]
fn duplicated_lower_knot_matches_numpy() {
    // np.interp(0, [0, 0, .5], [10, 5, 0]) == 5
    assert_eq!(interp1(0.0, &[0.0, 0.0, 0.5], &[10.0, 5.0, 0.0]), 5.0);
    assert_eq!(interp1(0.0, &[0.0, 0.0, 0.0, 0.5], &[10.0, 5.0, 3.0, 0.0]), 3.0);
}

#[test]
fn other_knots_unchanged() {
    // below the lower knot clamps; interior and upper duplicates take
    // the last; strictly ascending knots interpolate as before
    assert_eq!(interp1(-1.0, &[0.0, 0.0, 0.5], &[10.0, 5.0, 0.0]), 10.0);
    assert_eq!(interp1(1.0, &[0.0, 1.0, 1.0, 2.0], &[0.0, 5.0, 7.0, 9.0]), 7.0);
    assert_eq!(interp1(2.0, &[0.0, 2.0, 2.0], &[0.0, 5.0, 7.0]), 7.0);
    assert_eq!(interp1(0.0, &[0.0, 1.0, 2.0], &[4.0, 5.0, 7.0]), 4.0);
    assert_eq!(interp1(0.5, &[0.0, 1.0, 2.0], &[4.0, 5.0, 7.0]), 4.5);
    assert_eq!(interp1(3.0, &[0.0, 1.0, 2.0], &[4.0, 5.0, 7.0]), 7.0);
}

#[test]
fn compact_five_runner_calibration_reaches_target() {
    let mut density = vec![0.0; 25];
    density[9..16].copy_from_slice(&[0.0625, 0.125, 0.1875, 0.25, 0.1875, 0.125, 0.0625]);
    let truth = [-0.3870526359606288, -3.8859224668266945, -0.5618939753478802,
                 -6.990429141603045, 0.6690584211927497];
    let target = exact_state_prices_from_offsets(&density, &truth);
    let samples: Vec<f64> = (-10..=10).rev().map(|k| k as f64).collect();   // L = 12
    let implied = exact_calibrate(&density, &target, &samples, &[0.0; 5], 3);
    let back = exact_state_prices_from_offsets(&density, &implied);
    let err = target.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(err < 1e-5, "max reprice error {err}");             // was 0.0152
}
