/// Algorithms from or solving problems from the lecture *Algorithmen und Datenstrukturen 2*
/// at *Technische Universität Braunschweig* and/or its exercise sheets.
pub mod aud2 {
    use std::iter;

    /// **Homework #2 Task #1**
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

    /// **Homework #2 Task #1**
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

    /// **Homework #2 Task #1**
    /// An optimal **and** efficient algorithm for solving the CHANGE problem.
    pub fn change_optimal_dynamic(w: u128, m: &Vec<u128>) -> Vec<u128> {
        opt_with_result(m.len(), w, m).1
    }

    /// **Homework #2 Task #1**
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

    /// **Homework #2 Task #1**
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

    /// Helper function for the `CHANGE` problem and functions
    /// [change_greedy], [change_optimal_brute_force], [opt] and [opt_with_result].
    /// Panics whenever a constraint for the parameter `m` to the `CHANGE` problem (which is
    /// the list of available coins) is violated. These contraints are:
    /// * `m[0] == 1` (i.e. there has to be a `1` coin)
    /// * `m[0] < m[1] < m[2] < ...`
    fn panic_for_invalid_m(m: &Vec<u128>) {
        if m.len() == 0 {
            panic!("m.len() == 0");
        } else if m[0] != 1 {
            panic!("m[0] == 1 constraint violated");
        } else if !m.iter().fold((true, 0u128), |(is_sorted, prev_value), &value| (is_sorted && prev_value < value, value)).0 {
            panic!("m is not sorted");
        }
    }

    /// **Skript S.11, Algorithmus 2.1 "Dynamic Programming fuer SUBSET SUM"**
    ///
    /// Solves the `SUBSET SUM` problem (optimally) using *dynamic programming*.
    ///
    /// The `SUBSET SUM` problem is defined as follows:
    /// Let there be `n` objects `0,...,n-1`, each with size `z_i[0],...,z_i[n-1]`.
    /// Can we find a subset `S` of the set `{0,...,n-1}` such that the sum of all `z_i[i]` for all
    /// `i` in `S` is exactly equal to a given destination value `z`?
    ///
    /// This algorithm creates and returns a table in order to solve this problem.
    /// `result[x][i]` says whether the number `x` can be created using the first `i` values in `z_i`
    /// (both `x` and `i` begin at `0`).
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
    ///
    /// Transposes the given matrix `m`.
    ///
    /// Used to display the result of [subset_sum] to stdout in the same format
    /// that is used in the lecture.
    pub fn matrix_transpose<T>(m: Vec<Vec<T>>) -> Vec<Vec<T>> where T: Copy {
        let mut t = vec![Vec::with_capacity(m.len()); m[0].len()];
        for r in m {
            for i in 0..r.len() {
                t[i].push(r[i]);
            }
        }
        t
    }

    /// **Homework #3 Task #1**
    ///
    /// Solves the `MAXIMUM KNAPSACK` problem **optimally** using the *branch and bound* strategy
    /// (see [maximum_knapsack_greedy0_lower_bound] for a definition of the `MAXIMUM KNAPSACK` problem
    /// and the definitions of `small_z`, `big_z` and `small_p`).
    ///
    /// Returns the maximum possible total value of items that can be achieved by putting items
    /// into the knapsack of capacity `big_z`.
    ///
    /// **Note:** During computation, i.e. branching, the intermediate results will be
    /// printed to stdout in a tree-like manner.
    ///
    /// **Note:** This is a recursive function. Initially, `big_p` has to be set to `0`,
    /// `recursion_depth_l` to `1`
    /// and `small_b` `&vec![]`.
    /// `big_p` is the best solution known so far, `l` is the next index to branch over and
    /// `small_b` describes the binary decisions made/fixated so far, i.e. the current position
    /// in the tree.
    pub fn branch_and_bound_maximum_knapsack(small_z: &Vec<u128>, big_z: u128,
                                             small_p: &Vec<u128>, big_p: u128,
                                             recursion_depth_l: u128, small_b: &Vec<u8>) -> u128 {
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
        let mut big_p = big_p;
        let n = small_z.len();
        assert_eq!(n, small_p.len());

        if small_b.iter().zip(small_z.iter()).map(|(b_j, z_j)| (*b_j as u128)*z_j).sum::<u128>() > big_z {
            println!("{}{:?} is not allowed", "\t".repeat((recursion_depth_l-1) as usize), small_b);
            return big_p; // not allowed
        }
        let big_l = maximum_knapsack_greedy0_lower_bound(small_z, big_z, small_p, small_b);
        if big_l > big_p {
            big_p = big_l;
        }
        if recursion_depth_l > (n as u128) {
            println!("{}reached leaf {:?}", "\t".repeat((recursion_depth_l-1) as usize), small_b);
            return big_p; // reached leaf
        }
        let big_u = fractional_knapsack(small_z, big_z, small_p, small_b); // upper bound
        let big_u = big_u.floor() as u128; // the upper bound can be rounded down when it's fractional
        println!("{}U = {}", "\t".repeat((recursion_depth_l-1) as usize), big_u);
        println!("{}P = {}", "\t".repeat((recursion_depth_l-1) as usize), big_p);
        if big_u > big_p {
            let big_p_branch_left = branch_and_bound_maximum_knapsack(
                small_z, big_z,small_p, big_p,
                recursion_depth_l+1,
                &small_b.clone().into_iter().chain(iter::once(0u8)).collect());
            let big_p_branch_right = branch_and_bound_maximum_knapsack(
                small_z, big_z,small_p, big_p,
                recursion_depth_l+1,
                &small_b.clone().into_iter().chain(iter::once(1u8)).collect());
            big_p = big_p.max(big_p_branch_left).max(big_p_branch_right);
        }
        return big_p;
    }


    /// Solves the `MAXIMUM KNAPSACK` problem **non-optimally** using a *greedy algorithm*.
    ///
    /// The `MAXIMUM KNAPSACK` problem is defined as follows:
    ///
    /// Let there be a knapsack of capacity `big_z` and let there be `n` objects, each
    /// of size `small_z[0], ..., small_z[n-1]` and of value `small_p[0], ..., small_p[n-1]`.
    /// What is the maximum total value we can achieve by putting objects into the knapsack
    /// but still obeying the maximum capacity `big_z`?
    ///
    /// **Important:** Unlike `FRACTIONAL KNAPSACK`, `MAXIMUM KNAPSACK` does **not** allow objects
    /// to be put into the knapsack partially/to any arbitrary fraction! Every objects either has
    /// to be taken as a whole or not at all!
    ///
    /// **Note:** When the `fixated_small_b` given is not the empty vector `vec![]`, the `n`-th
    /// object will be definitely *not* taken when `fixated_small_b[n-1] == 0` and it *will* be
    /// definitely taken when `fixated_small_b[n-1] == 1`.
    ///
    /// **Note:** This function gives a **lower bound** for the `MAXIMUM KNAPSACK`
    /// problem as it is a naive and not necessarily optimal solution for it.
    pub fn maximum_knapsack_greedy0_lower_bound(small_z: &Vec<u128>, big_z: u128,
                                                small_p: &Vec<u128>, fixated_small_b: &Vec<u8>) -> u128 {
        /* ====== HA-Blatt 1: ======
        1: function Greedy0(z1,...,zn,Z,p1,...,pn)
        2:     Sortiere Objekte nach z[i]/p[i] aufsteigend; dies ergibt Permutation π(1),...,π(n).
        3:     for j := 1 to n do
        4:         if SIGMA(i=1,j-1) x[π(i)]*z[π(i)] <= Z then
        5:             x[π(j)] := 1
        6:         else
        7:             x[π(j)] := 0
        8:      return x[1],...,x[n]
         */
        let n = small_z.len();
        assert_eq!(n, small_p.len());

        //let mut result_x: Vec<u8> = iter::repeat(0).take(n).collect(); // without fixation
        // with fixation:
        let fixated_small_b_len = fixated_small_b.len();
        let mut result_x: Vec<u8> = fixated_small_b
            .clone()
            .into_iter()
            .chain(iter::repeat(0).take(n-fixated_small_b_len))
            .collect();
        let mut objects: Vec<(u128, u128)> = small_z.clone().into_iter().zip(small_p.clone().into_iter()).collect();
        objects.sort_by(|(z_i1, p_i1), (z_i2, p_i2)|
                            ((*z_i1 as f64) / (*p_i1 as f64)).partial_cmp(&((*z_i2 as f64) / (*p_i2 as f64)))
                                .unwrap()); // unwrap because partial_cmp returns an Option  // ToDo: unnecessary sorting?!

        let sorted_small_z: Vec<u128> = objects.iter().map(|(z_i, _p_i)| *z_i).collect(); // ToDo: unnecessary sorting?!
        let sorted_small_p: Vec<u128> = objects.iter().map(|(_z_i, p_i)| *p_i).collect(); // ToDo: unnecessary sorting?!

        for j in fixated_small_b_len..n { //for j in 0..n { // without fixation
            if j<n && result_x.iter().zip(sorted_small_z.iter()).map(|(x_i, z_i)| (*x_i as u128)*z_i).sum::<u128>() + 1*(*sorted_small_z.iter().nth(j).unwrap()) <= big_z {
                result_x[j] = 1;
            } else {
                result_x[j] = 0;
            }
        }
        return result_x.iter().zip(sorted_small_p.iter()).map(|(x_i, p_i)| (*x_i as u128)*p_i).sum::<u128>(); // see above
    }

    /// Solves the `FRACTIONAL KNAPSACK` problem optimally using a *greedy algorithm*.
    ///
    /// The `FRACTIONAL KNAPSACK` problem is defined as follows:
    ///
    /// Let there be a knapsack of capacity `big_z` and let there be `n` objects, each
    /// of size `small_z[0], ..., small_z[n-1]` and of value `small_p[0], ..., small_p[n-1]`.
    /// What is the maximum total value we can achieve by putting objects into the knapsack
    /// but still obeying the maximum capacity `big_z`?
    ///
    /// **Important:** `FRACTIONAL KNAPSACK` allows objects to be put into the knapsack partially,
    /// i.e. to any arbitrary fraction!
    ///
    /// **Note:** When the `fixated_small_b` given is not the empty vector `vec![]`, the `n`-th
    /// object will be definitely *not* taken when `fixated_small_b[n-1] == 0` and it *will* be
    /// definitely taken when `fixated_small_b[n-1] == 1`. In this case,
    /// the function does not return the optimal solution for the total problem but rather
    /// for the problem with some objects fixed, as defined in the given `fixated_small_b` vector.
    ///
    /// **Note:** This function gives an **upper bound** for the more strict `MAXIMUM KNAPSACK`
    /// problem which does not allow objects to be taken partially.
    pub fn fractional_knapsack(small_z: &Vec<u128>, big_z: u128,
                               small_p: &Vec<u128>, fixated_small_b: &Vec<u8>) -> f64 {
        /*
        ====== Skript S.3: Algorithmus 1.4 Greedy-Algorithmus für Fractional Knapsack ======
        Eingabe: z[1],...,z[n],Z,p[1],...,p[n]
        Ausgabe: x[1],...,x[n] ∈[0,1]
        mit SIGMA(i=1,n) z[i]*x[i] <= Z
        und SIGMA(i=1,n) p[i]*x[i] = Maximal

        1: Sortiere {1,...,n} nach z[i]/p[i] aufsteigend; Dies ergibt die Permutation π(1),...,π(n).
           Setze j = 1.
        2: while (SIGMA(i=1,j) z[π(i)] <= Z) do
        3:     x[π(j)] := 1
        4:     j := j + 1
        5: Setze x[π(j)] := (Z - SIGMA(i=1,j-1) z[π(i)] ) / z[π(j)]
        6: return
         */
        let n = small_z.len();
        assert_eq!(n, small_p.len());

        /* Copied from maximum_knapsack_greedy0_lower_bound() function above: */
        let fixated_small_b_len = fixated_small_b.len();
        let mut result_x: Vec<f64> = fixated_small_b
            .into_iter()
            .map(|&x| x as f64)
            .chain(iter::repeat(0f64).take(n-fixated_small_b_len))
            .collect();
        let mut objects: Vec<(u128, u128)> = small_z.clone().into_iter().zip(small_p.clone().into_iter()).collect();
        objects.sort_by(|(z_i1, p_i1), (z_i2, p_i2)|
            ((*z_i1 as f64) / (*p_i1 as f64)).partial_cmp(&((*z_i2 as f64) / (*p_i2 as f64)))
                .unwrap()); // unwrap because partial_cmp returns an Option  // ToDo: unnecessary sorting?!
        /* */
        //println!("objects = {:?}", objects); // for bug-fixing
        let sorted_small_z: Vec<u128> = objects.iter().map(|(z_i, _p_i)| *z_i).collect(); // ToDo: unnecessary sorting?!
        let sorted_small_p: Vec<u128> = objects.iter().map(|(_z_i, p_i)| *p_i).collect(); // ToDo: unnecessary sorting?!

        let mut j = fixated_small_b_len; // let mut j = 0; // without fixation
        while j<n && result_x.iter().zip(sorted_small_z.iter()).map(|(x_i, z_i)| x_i*(*z_i as f64)).sum::<f64>() + 1.0*(*sorted_small_z.iter().nth(j).unwrap() as f64) <= (big_z as f64) {
            result_x[j] = 1.0;
            j += 1;
        }
        if j<n {
            result_x[j] = ((big_z as f64) - result_x.iter().zip(sorted_small_z.iter()).map(|(x_i, z_i)| x_i * (*z_i as f64)).sum::<f64>())
                / (*sorted_small_z.iter().nth(j).unwrap() as f64); // Setze x[π(j)] := (Z - SIGMA(i=1,j-1) z[π(i)] ) / z[π(j)]
        }
        //println!("result_x = {:?}", result_x); // for bug-fixing
        return result_x.iter().zip(sorted_small_p.iter()).map(|(x_i, p_i)| x_i*(*p_i as f64)).sum::<f64>();
    }

    /// **Homework #3 Task #2a (Weighted Lecture Hall Problem)**
    ///
    /// Returns `[p(0), p(1), p(2), ..., p(n)]` as a vector of `n+1` elements where:
    /// * `n == s.len() == e.len()`
    /// * `(s[0], e[0]), ..., (s[n-1], e[n-1])` are intervals with `e[0] <= e[2] <= ... <= e[n-1]`
    /// * `p(0) := 0`
    /// * `p(i) := max({j | e[j-1] < s[i-1]} + {0})`, i.e. `p(i)` is the largest index of the interval
    ///   that ends the latest but still before interval number `i` begins.
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
            // For my Rust implementation it has to be "<" and not "<="
            //   because in my test cases, I interpret the intervals intuitively
            //   as closed intervals [s, e].
            //   The official task however understands them as open intervals (s, e).
            //   Therefore, the official solution requires a "<=" instead of a "<"!
            while e[j-1] < s[i-1].1 {
                j += 1;
            }
            j -= 1;
            p[s[i-1].0 + 1] = j;
            j += 1;
        }

        return p;
    }

    /// **Homework #3 Task #2b (Weighted Lecture Hall Problem)**
    ///
    /// Solves the *Weighted Lecture Hall Problem*:
    ///
    /// When `(s[0], e[0]), ..., (s[n-1], e[n-1])` are intervals with
    /// `e[0] <= e[2] <= ... <= e[n-1]` and weights `w[0], ..., w[n-1]`, then `G(j)` is the largest
    /// possible value of the sum of all `w[i]` for all `i` in a set `S`
    /// which is a subset of disjoint intervals, i.e. a subset of `{0,...,j-1}`.
    ///
    /// This function returns `G(n)`.
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

    /// Helper function for `ps()`:
    ///
    /// Checks for a given vector `v` whether it is sorted in ascending order (non-strict), i.e.
    /// whether `v[0] <= v[1] <= ... <= v[v.len()-1]`
    /// Returns false if and only if there is an index `i` such that `v[i] > v[i+1]`.
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
        todo!()
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
    fn test_branch_and_bound_maximum_knapsack() {
        // Example from Homework #3:
        let result = branch_and_bound_maximum_knapsack(
            &vec![18, 19, 9, 13], 32,
            &vec![21, 19, 8, 11], 0, // best known solution is initially 0
            1, &vec![] // initial recursion depth is 1 and nothing is fixed yet
        );
        assert_eq!(result, 32); // 32 should be the best result by taking the elements #1 and #4

        println!();
        println!();
        println!();

        // Example from U3:
        let result = branch_and_bound_maximum_knapsack(
            &vec![11, 5, 13, 18, 9], 33,
            &vec![14, 6, 13, 16, 7], 0,
            1, &vec![]
        );
        assert_eq!(result, 34)
    }

    #[test]
    fn test_branch_and_bound_maximum_knapsack_lower_bound() {
        // maximum_knapsack_greedy0_lower_bound(small_z: &Vec<u128>, big_z: u128,
        //                                      small_p: &Vec<u128>, fixated_small_b: &Vec<u8>) -> u128

        // Blatt #3 Aufgabe #1:
        assert_eq!(29, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![]));
        // 1) fix b1=0:
        assert_eq!(27, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![0]));
        // 2) fix b1=0 and b2=0:
        assert_eq!(19, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![0, 0]));
        // 3) fix b1=0, b2=1 and b3=0:
        assert_eq!(30, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![0, 1, 0]));
        // 4) fix b1=1 and b2=0:
        assert_eq!(29, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![1, 0]));
        // 5) fix b1=1, b2=0 and b3=0:
        assert_eq!(32, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![1, 0, 0]));
        // 6) fix b1=1, b2=0 and b3=1: (== 4))
        assert_eq!(29, maximum_knapsack_greedy0_lower_bound(&vec![18, 19, 9, 13], 32,
                                                            &vec![21, 19, 8, 11], &vec![1, 0, 1]));
    }

    #[test]
    fn test_branch_and_bound_maximum_knapsack_upper_bound() {
        // fractional_knapsack(small_z: &Vec<u128>, big_z: u128,
        //                     small_p: &Vec<u128>, fixated_small_b: &Vec<u8>) -> f64

        // Blatt #3 Aufgabe #1:
        assert_eq!(35.0, fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                             &vec![21, 19, 8, 11], &vec![]));

        // 1) fix b1=0:
        assert!((30.0+5.0/13.0 - fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                                      &vec![21, 19, 8, 11], &vec![0])
        ).abs() < 0.00000001);

        // 2) fix b1=0 and b2=0:
        assert_eq!(19.0, fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                             &vec![21, 19, 8, 11], &vec![0, 0]));

        // 3) fix b1=0, b2=1 and b3=0:
        assert_eq!(30.0, fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                             &vec![21, 19, 8, 11], &vec![0, 1, 0]));

        // 4) fix b1=1 and b2=0:
        assert!((33.2308 - fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                             &vec![21, 19, 8, 11], &vec![1, 0])
        ).abs() < 0.001);

        // 5) fix b1=1, b2=0 and b3=0:
        assert_eq!(32.0, fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                             &vec![21, 19, 8, 11], &vec![1, 0, 0]));

        // 6) fix b1=1, b2=0 and b3=1: (== 4))
        assert!((33.2308 - fractional_knapsack(&vec![18, 19, 9, 13], 32,
                                             &vec![21, 19, 8, 11], &vec![1, 0, 1])
        ).abs() < 0.001);
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