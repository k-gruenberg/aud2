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

    /// Homework #2 Task #1
    /// An optimal **and** efficient algorithm for solving the CHANGE problem.
    pub fn change_optimal_dynamic(w: u128, m: &Vec<u128>) -> Vec<u128> {
        opt_with_result(m.len(), w, m).1
    }

    /// Homework #2 Task #1
    /// Returns the minimum **amount** of coins needed, taken from the first `i` coins in `m`,
    /// to add up to exactly `x`.
    /// As the parameter `i` suggests, this function is defined recursively.
    /// This function will **not** tell you which coins to actually choose from `m`, only how many
    /// in total you have to take.
    pub fn opt(i: usize, x: u128, m: &Vec<u128>) -> u128 {
        // "Sei nun opt(i, x) der minimale Wert, wie viele der ersten i Münzen benötigt werden,
        //  um den Wert x zu erreichen [...] für ein Währungssystem mit Münzen im Wert von"
        //  1 = m[0] < m[1] < · · · < m[m.len()-1]

        panic_for_invalid_m(&m);

        match (i,x) {
            (0,0) => 0, // 0 coins from the set {} are needed to reach the value of 0
            (0,_) => u128::MAX, // it is impossible to reach a value >0 by using coins from the set {}
            (1,x) => x, // x coins from the set {1} are needed to reach the value of x
            (i,x) => {
                // Use the i-th coin n-times, i.e. 0 times or 1 time or ... or x/m[i-1] times
                //   (do the rest using the other coins 1 to i-1);
                //   then take the best, i.e. minimum:
                (0..=x/m[i-1])
                    .map(|n| n + opt(i-1, x - n*m[i-1], m))
                    .min()
                    .expect("iterator always contains at least 1 element")
            }
        }
    }

    /// Homework #2 Task #1
    /// Does the same as the `opt()` function, but it also keeps track of the result as a vector
    /// which is returned.
    /// It might be more convenient to use the `change_optimal_dynamic()` function however!
    pub fn opt_with_result(i: usize, x: u128, m: &Vec<u128>) -> (u128, Vec<u128>) {
        panic_for_invalid_m(&m);

        match (i,x) {
            (0,0) => (0, iter::repeat(0).take(m.len()).collect()), // 0 coins from the set {} are needed to reach the value of 0
            (0,_) => (u128::MAX, Vec::new()), // it is impossible to reach a value >0 by using coins from the set {}
            (1,x) => (x, iter::once(x).chain(iter::repeat(0)).take(m.len()).collect()), // x coins from the set {1} are needed to reach the value of x
            (i,x) => {
                // Use the i-th coin n-times, i.e. 0 times or 1 time or ... or x/m[i-1] times
                //   (do the rest using the other coins 1 to i-1);
                //   then take the best, i.e. minimum:
                (0..=x/m[i-1])
                    .map(|n| {
                        let (amount, mut coins) = opt_with_result(i-1, x - n*m[i-1], m);
                        coins[i-1] += n; // tool the i-th coin (another) n-times
                        (n + amount, coins)
                    })
                    .min_by_key(|(amount, _coins)| *amount) // (or amount.clone())
                    .expect("iterator always contains at least 1 element")
            }
        }
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

    /// Skript S.11, Algorithmus 2.1 "Dynamic Programming fuer SUBSET SUM"
    pub fn subset_sum(z_i: &Vec<usize>, z: usize) -> Vec<Vec<u8>> {
        let n = z_i.len();
        let mut result: Vec<Vec<u8>> = iter::repeat(iter::repeat(0).take(n+1).collect()).take(z+1).collect();
        result[0][0] = 1;
        for x in 1..=z {
            result[x][0] = 0;
        }
        for i in 1..=n {
            for x in 0..=z_i[i-1]-1 {
                result[x][i] = result[x][i-1];
            }
            for x in z_i[i-1]..=z {
                if result[x][i-1] == 1 || result[x-z_i[i-1]][i-1] == 1 {
                    result[x][i] = 1;
                } else {
                    result[x][i] = 0;
                }
            }
        }
        return result;
    }

    /// Helper function, copied from: https://www.hackertouch.com/matrix-transposition-in-rust.html
    pub fn matrix_transpose<T>(m: Vec<Vec<T>>) -> Vec<Vec<T>> where T: Copy {
        let mut t = vec![Vec::with_capacity(m.len()); m[0].len()];
        for r in m {
            for i in 0..r.len() {
                t[i].push(r[i]);
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use crate::aud2::*;

    #[test]
    fn test_change() {
        //todo!()
    }

    #[test]
    fn test_opt_function() {
        let coins_without_four = vec![1,2,5,10,20,50,100,200];
        let coins_with_four = vec![1,2,4,5,10,20,50,100,200];

        // try to add up coins to the total value of $8, with and without a $4 coin:
        assert_eq!(opt(coins_without_four.len(), 8, &coins_without_four), 3);
        assert_eq!(opt(coins_with_four.len(), 8, &coins_with_four), 2);

        // Without the use of a $4 coin, the opt() function should return the same as the greedy approach:
        // 177 = 100 + 50 + 20 + 5 + 2
        assert_eq!(opt(coins_without_four.len(), 177, &coins_without_four), 5);
        // 178 = 100 + 50 + 20 + 5 + 2 + 1
        assert_eq!(opt(coins_without_four.len(), 178, &coins_without_four), 6);
        // 44 = 20 + 20 + 2 + 2
        assert_eq!(opt(coins_without_four.len(), 44, &coins_without_four), 4);
        // 3 = 2 + 1
        assert_eq!(opt(coins_without_four.len(), 3, &coins_without_four), 2);
        // 50 = 50
        assert_eq!(opt(coins_without_four.len(), 50, &coins_without_four), 1);
    }

    #[test]
    fn test_opt_with_result_function() {
        let coins_without_four = vec![1,2,5,10,20,50,100,200];
        let coins_with_four = vec![1,2,4,5,10,20,50,100,200];

        // try to add up coins to the total value of $8, with and without a $4 coin:
        assert_eq!(opt_with_result(coins_without_four.len(), 8, &coins_without_four).0, 3);
        assert_eq!(opt_with_result(coins_without_four.len(), 8, &coins_without_four).1, vec![1,1,1,0,0,0,0,0]);
        assert_eq!(opt_with_result(coins_with_four.len(), 8, &coins_with_four).0, 2);
        assert_eq!(opt_with_result(coins_with_four.len(), 8, &coins_with_four).1, vec![0,0,2,0,0,0,0,0,0]);

        // Without the use of a $4 coin, the opt() function should return the same as the greedy approach:
        // 177 = 100 + 50 + 20 + 5 + 2
        assert_eq!(opt_with_result(coins_without_four.len(), 177, &coins_without_four).0, 5);
        assert_eq!(opt_with_result(coins_without_four.len(), 177, &coins_without_four).1, vec![0,1,1,0,1,1,1,0]);
        // 178 = 100 + 50 + 20 + 5 + 2 + 1
        assert_eq!(opt_with_result(coins_without_four.len(), 178, &coins_without_four).0, 6);
        assert_eq!(opt_with_result(coins_without_four.len(), 178, &coins_without_four).1, vec![1,1,1,0,1,1,1,0]);
        // 44 = 20 + 20 + 2 + 2
        assert_eq!(opt_with_result(coins_without_four.len(), 44, &coins_without_four).0, 4);
        assert_eq!(opt_with_result(coins_without_four.len(), 44, &coins_without_four).1, vec![0,2,0,0,2,0,0,0]);
        // 3 = 2 + 1
        assert_eq!(opt_with_result(coins_without_four.len(), 3, &coins_without_four).0, 2);
        assert_eq!(opt_with_result(coins_without_four.len(), 3, &coins_without_four).1, vec![1,1,0,0,0,0,0,0]);
        // 50 = 50
        assert_eq!(opt_with_result(coins_without_four.len(), 50, &coins_without_four).0, 1);
        assert_eq!(opt_with_result(coins_without_four.len(), 50, &coins_without_four).1, vec![0,0,0,0,0,1,0,0]);
    }
}