use aud2::aud2::change_greedy;
use aud2::aud2::change_optimal_brute_force;
use aud2::aud2::change_optimal_dynamic;

fn main() {
    let coins = vec![1,2,4,5,10,20,50,100,200];

    println!("w\t\t\tGreedy\t\tBrute-f\t\tDynamic");

    // brute-force = (x_max+1).pow(m.len())
    //             = (w / m.iter().min().unwrap() + 1).pow(m.len())
    // for w=5 already:
    //             = (5 / 1 + 1).pow(9)
    //             = 6.pow(9)
    //             = 10 077 696
    const BRUTE_FORCE_MAX: u128 = 8;

    for w in 0..100 {
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
