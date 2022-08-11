use aud2::aud2::{branch_and_bound_maximum_knapsack, change_greedy};
use aud2::aud2::change_optimal_brute_force;
use aud2::aud2::change_optimal_dynamic;
use aud2::aud2::subset_sum;
use aud2::aud2::matrix_transpose;

fn main() {
    println!("{}",
        branch_and_bound_maximum_knapsack(&vec![50,5,5,50,10,20,10], 56,
                                      &vec![100,10,10,50,10,5,1], 0,
                                          1, &vec![])
    );
    return;

    for vector in matrix_transpose(subset_sum(&vec![7usize,4,1,9,3], 15)) {
        println!("{:?}", vector);
    }
    println!();
    println!();
    println!();



    let coins = vec![1,2,5,10,20,40,50,100,200]; // vec![1,2,4,5,10,20,50,100,200];

    println!("w\t\t\tGreedy\t\tBrute-f\t\tDynamic");

    // brute-force = (x_max+1).pow(m.len())
    //             = (w / m.iter().min().unwrap() + 1).pow(m.len())
    // for w=5 already:
    //             = (5 / 1 + 1).pow(9)
    //             = 6.pow(9)
    //             = 10 077 696
    const BRUTE_FORCE_MAX: u128 = 6; // =8

    for w in vec![11,80,562,1123] { // 0..10
        let greedy_solution = change_greedy(w, &coins);
        let optimal_brute_force_solution =
            if w <= BRUTE_FORCE_MAX {Some(change_optimal_brute_force(w, &coins))} else {None};
        let optimal_dynamic_solution = change_optimal_dynamic(w, &coins);

        print!("{}\t\t\t{}\t\t\t{}\t\t\t{}",
               w,
               greedy_solution.iter().sum::<u128>(),
               optimal_brute_force_solution.as_ref().map(|sol| sol.iter().sum::<u128>().to_string()).unwrap_or("?".to_string()),
               optimal_dynamic_solution.iter().sum::<u128>()
        );

        // Add additional columns to a row iff Greedy does not give an optimal solution:
        if greedy_solution.iter().sum::<u128>() > optimal_dynamic_solution.iter().sum::<u128>() {
            println!("\t\t\t{:?}\t\t\t{:?}\t\t\t{:?}",
                     greedy_solution,
                     optimal_brute_force_solution.unwrap_or(vec![]),
                     optimal_dynamic_solution
            );
        } else {
            println!();
        }
    }
}
