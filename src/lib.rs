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
            x[i] = w / m[i]; // (pseudo-code: "x[i] := floor(w / m[i])")
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
        for _ in 0..(x_max + 1).pow(m.len() as u32) {
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

            if current_x.iter().zip(m.iter()).map(|(&xi, &mi)| xi * mi).sum::<u128>() == w // the current vector x results in the exact change we need
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

        match (i, x) {
            (0, 0) => 0, // 0 coins from the set {} are needed to reach the value of 0
            (0, _) => u128::MAX, // it is impossible to reach a value >0 by using coins from the set {}
            (1, x) => x, // x coins from the set {1} are needed to reach the value of x
            (i, x) => {
                // Use the i-th coin n-times, i.e. 0 times or 1 time or ... or x/m[i-1] times
                //   (do the rest using the other coins 1 to i-1);
                //   then take the best, i.e. minimum:
                (0..=x / m[i - 1])
                    .map(|n| n + opt(i - 1, x - n * m[i - 1], m))
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

        match (i, x) {
            (0, 0) => (0, iter::repeat(0).take(m.len()).collect()), // 0 coins from the set {} are needed to reach the value of 0
            (0, _) => (u128::MAX, Vec::new()), // it is impossible to reach a value >0 by using coins from the set {}
            (1, x) => (x, iter::once(x).chain(iter::repeat(0)).take(m.len()).collect()), // x coins from the set {1} are needed to reach the value of x
            (i, x) => {
                // Use the i-th coin n-times, i.e. 0 times or 1 time or ... or x/m[i-1] times
                //   (do the rest using the other coins 1 to i-1);
                //   then take the best, i.e. minimum:
                (0..=x / m[i - 1])
                    .map(|n| {
                        let (amount, mut coins) = opt_with_result(i - 1, x - n * m[i - 1], m);
                        coins[i - 1] += n; // tool the i-th coin (another) n-times
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
        let mut result: Vec<Vec<u8>> = iter::repeat(iter::repeat(0).take(n + 1).collect()).take(z + 1).collect();
        result[0][0] = 1;
        for x in 1..=z {
            result[x][0] = 0;
        }
        for i in 1..=n {
            for x in 0..=z_i[i - 1] - 1 {
                result[x][i] = result[x][i - 1];
            }
            for x in z_i[i - 1]..=z {
                if result[x][i - 1] == 1 || result[x - z_i[i - 1]][i - 1] == 1 {
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

    /// Homework #3 Task #1
    pub fn branch_and_bound_maximum_knapsack(recursion_depth: u128) -> () {
        /* ===== ===== Skript, S. 18: ===== =====
        ===== Algorithmus 3.1 Branch-And-Bound als Unterroutine =====
        Eingabe: z[1],...,z[n],Z,p[1],...p[n] (global:Kosten/Nutzenwerte,Kostenschranke)
                 P                    (bester bekannter Lösungswert)
                 l                    (nächster Index, über den verzweigt werden soll)
                 x[j] = b[j] für j = 1,...,l−1 mit b[j] ∈ {0,1} (bislang fixierte Binärvariable)
        Ausgabe: max {SIGMA(j=1,l-1) b[j]*p[j] + SIGMA(j=l,n) x[j]*p[j] |
                      SIGMA(j=1,l-1) b[j]*z[j] + SIGMA(j=l,n) x[j]*z[j] <= Z, x[j] ∈ {0,1}}
        Also: Lösung des Knapsackproblems mit den ersten l−1 Variablen fixiert

        1: procedure Branch-and-Bound(l)
        2:     if ( SIGMA(j=1,l-1) b[j]*z[j] > Z ) then return // unzulässig
        3:     Compute L := LB(b[1],...,b[l−1])
        4:     if L > P then P := L               // Lösungswert verbessert
        5:     if (l > n) then return             // Blatt im Baum erreicht
        6:     U := UB(b[1],...,b[l−1])           // (Obere Schranke berechnen)
        7:     if (U>P) then
        8:         b[l] := 0; Branch-and-Bound(l+1);
        9:         b[l] := 1; Branch-and-Bound(l+1);
        10:    return
         */
    }

    /// Homework #3 Task #2a (Weighted Lecture Hall Problem)
    pub fn ps(s: Vec<u128>, e: Vec<u128>) -> Vec<usize> {
        let n = s.len();
        assert_eq!(n, e.len());
        assert!(is_sorted(&e), "e={:?} is not sorted", &e); // assert!(e.is_sorted()); // unstable Rust

        let mut s: Vec<(usize, u128)> = s.into_iter().enumerate().collect();
        s.sort_by_key(|(_i, s_i)| *s_i);

        let mut p: Vec<usize> = iter::repeat(0).take(n+1).collect();
        // p[0] should remain unaltered throughout the program below (it's undefined):

        let mut j = 1;
        for i in 1..=n {
            while e[j-1] <= s[i-1].1 {
                j += 1;
            }
            j -= 1;
            p[s[i-1].0 + 1] = j;
            j += 1;
        }

        return p;
    }

    /// Homework #3 Task #2b (Weighted Lecture Hall Problem)
    pub fn compute_g(s: Vec<u128>, e: Vec<u128>, w: Vec<u128>) -> u128 {
        let n = s.len();
        assert_eq!(n, e.len());
        assert_eq!(n, w.len());

        let p = ps(s, e);

        let mut g: Vec<u128> = iter::repeat(0).take(n+1).collect();

        for i in 1..=n {
            g[i] = g[i-1].max(g[p[i]] + w[i-1]);
        }

        return g[n];
    }

    /// Helper function for ps():
    fn is_sorted<T>(v: &Vec<T>) -> bool where T: Ord {
        for i in 0..v.len()-1 {
            if v[i] > v[i+1] {
                return false;
            }
        }
        return true;
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

    #[test]
    fn test_weighted_lecture_hall_problem() {
        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ====50====
            ====90====
                           ====5=====
        ================100=================
         */
        let s = vec![0, 4, 19, 0];
        let e = vec![9, 13, 28, 35];
        let w = vec![50, 90, 5, 100];
        //assert_eq!(ps(s.clone(), e.clone()), vec![0, ...]);
        assert_eq!(compute_g(s, e, w), 100);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ====50====
                    ====90====
                                ====5=====
        ================100=================
         */
        let s = vec![0, 12, 24, 0];
        let e = vec![9, 21, 33, 35];
        let w = vec![50, 90, 5, 100];
        assert_eq!(compute_g(s, e, w), 145);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ====50====
                  ====90====
                            ====5=====
        ================100=================
         */
        let s = vec![0, 10, 20, 0];
        let e = vec![9, 19, 29, 35];
        let w = vec![50, 90, 5, 100];
        assert_eq!(compute_g(s, e, w), 145);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ====50====
                         ====90====
                                ====5=====
        ================100=================
         */
        let s = vec![0, 17, 24, 0];
        let e = vec![9, 26, 33, 35];
        let w = vec![50, 90, 5, 100];
        assert_eq!(compute_g(s, e, w), 140);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ====50====
                 ====90====
                          ====5=====
        ================100=================
         */
        let s = vec![0, 9, 18, 0];
        let e = vec![9, 18, 27, 35];
        let w = vec![50, 90, 5, 100];
        assert_eq!(ps(s.clone(), e.clone()), vec![0, 0, 0, 1, 0]);
        //   "[...] gibt also den größten Index des Intervalls zurück, das am spätesten,
        //    aber noch vor Intervall I[i] endet"
        assert_eq!(compute_g(s, e, w), 100);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ====50====
                 ====5=====
                          ====90====
        ================100=================
         */
        let s = vec![0, 9, 18, 0];
        let e = vec![9, 18, 27, 35];
        let w = vec![50, 5, 90, 100];
        assert_eq!(compute_g(s, e, w), 140);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        =100=
                                        =99=
         */
        let s = vec![0, 32];
        let e = vec![4, 35];
        let w = vec![100, 99];
        assert_eq!(compute_g(s, e, w), 199);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        =100=
                                        =99=
        ================198=================
         */
        let s = vec![0, 32, 0];
        let e = vec![4, 35, 35];
        let w = vec![100, 99, 198];
        assert_eq!(compute_g(s, e, w), 199);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        =100=
                                        =99=
        ================199=================
         */
        let s = vec![0, 32, 0];
        let e = vec![4, 35, 35];
        let w = vec![100, 99, 199];
        assert_eq!(compute_g(s, e, w), 199);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        =100=
                                        =99=
        ================200=================
         */
        let s = vec![0, 32, 0];
        let e = vec![4, 35, 35];
        let w = vec![100, 99, 200];
        assert_eq!(compute_g(s, e, w), 200);

        /*
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
           ===100===
                   ====200===
         */
        let s = vec![3, 11];
        let e = vec![11, 20];
        let w = vec![100, 200];
        assert_eq!(compute_g(s, e, w), 200);

        // ========== Examples from U1: ==========

        /* A counter-example to the shortest-first strategy:
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
             ===1===
                   =1=
                     ===1==
         */
        let s = vec![5, 11, 13];
        let e = vec![11, 13, 18];
        let w = vec![1, 1, 1];
        assert_eq!(compute_g(s, e, w), 2);

        /* A counter-example to both the earliest-start and longest-interval strategies:
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
         =1=
            =1=
               =1=
                  =1=
                     =1=
                        =1=
                           =1=
        ==========1===========
         */
        let s = vec![1, 4, 7, 10, 13, 16, 19, 0];
        let e = vec![3, 6, 9, 12, 15, 18, 21, 21];
        let w = vec![1, 1, 1, 1,  1,  1,  1,  1];
        assert_eq!(compute_g(s, e, w), 7);

        /* A counter-example to the few-overlaps strategy:
        0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        ===1===
              =1=
              =1=
              =1=
                ===1===
                     ==1==
                        ===1===
                              =1=
                              =1=
                              =1=
                                ===1===
         */
        let s = vec![0, 6, 6, 6,  8, 13, 16, 22, 22, 22, 24];
        let e = vec![6, 8, 8, 8, 14, 17, 22, 24, 24, 24, 30];
        let w = vec![1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1];
        assert_eq!(compute_g(s, e, w), 4);
    }
}