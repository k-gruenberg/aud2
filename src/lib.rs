pub mod aud2 {
    use std::iter;

    /// Homework #2 Task #1
    /// The Greedy Algorithm solving the CHANGE problem, as given on the second exercise sheet.
    /// May or may not solve the CHANGE problem optimally, depending on the vector `m`!
    pub fn change_greedy(mut w: u128, m: &Vec<u128>) -> Vec<u128> { // (pseudo-code: "function Change(W, m[1], ..., m[small_m])")
        panic_for_invalid_m(&m);

        let small_m = m.len();
        let mut x: Vec<u128> = iter::repeat(0).take(small_m).collect();

        for i in (0..small_m).rev() { // (pseudo-code: "for i = m down to 1 do:")
            x[i] = w/m[i]; // (pseudo-code: "x[i] := floor(w / m[i])")
            w = w - x[i] * m[i] // (pseudo-code: "W := W - x[i] * m[i]")
        }

        return x; // (pseudo-code: "return x[1], ..., x[m]")
    }

    /// Homework #2 Task #1
    /// A very naive and **very inefficient** brute-force algorithm for solving the CHANGE problem optimally.
    pub fn change_optimal_brute_force(w: u128, m: &Vec<u128>) -> Vec<u128> {
        panic_for_invalid_m(&m);

        // The maximum value any `x[i]` may reasonably have is `w / min(m)`:
        let x_max = w / m.iter().min().expect("m is empty");
        // Having this value is crucial, otherwise brute-forcing would require testing *infinitely*
        // many possibilities for the values of the vector `x`!

        let mut current_x: Vec<u128> = iter::repeat(0).take(m.len()).collect();
        let mut current_best_x = Vec::new();

        let mut best_sum_so_far = u128::MAX;

        // Try out *all* possible vectors of length `m.len()` with values between `0` and `x_max`
        // (a total of `(x_max+1).pow(m.len())` possibilities!) for the vector `current_x`:
        for _ in 0..(x_max+1).pow(m.len() as u32) {
            // Interpret `current_x` as a `current_x.len()`-digit number in base `x_max+1`
            // and increment this number by one (this way we try out all possibilities for `current_x` with values <= `x_max`):
            for i in (0..current_x.len()).rev() { // (reversing isn't strictly necessary but it seems more natural)
                if current_x[i] == x_max {
                    current_x[i] = 0; // adding 1 rolls the digit over from `x_max` to `0`
                    // keep the carry of 1, i.e. continue in the loop
                } else {
                    current_x[i] += 1;
                    break; // no carry left: stop iterating.
                }
            }

            if current_x.iter().zip(m.iter()).map(|(&xi, &mi)| xi*mi).sum::<u128>() == w // the current vector x results in the exact change we need
                && current_x.iter().sum::<u128>() < best_sum_so_far { // the sum of the elements in vector x is smaller, i.e. better than our current best
                    current_best_x = current_x.clone(); // update current best vector x
                    best_sum_so_far = current_x.iter().sum::<u128>(); // update current best sum SIGMA(i)(x[i])
            }
        }

        return current_best_x;
    }

    fn panic_for_invalid_m(m: &Vec<u128>) {
        if m.len() == 0 {
            panic!("m.len() == 0");
        } else if m[0] != 1 {
            panic!("m[0] == 1 constraint violated");
        } else if !m.iter().fold((true, 0u128), |(is_sorted, prev_value), &value| (is_sorted && prev_value < value, value)).0 {
            panic!("m is not sorted");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::aud2::*;

    #[test]
    fn test_change() {
        todo!()
    }
}