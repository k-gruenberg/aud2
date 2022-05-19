use aud2::aud2::change_greedy;
use aud2::aud2::change_optimal_brute_force;

fn main() {
    let coins = vec![1,2,4,5,10,20,50,100,200];

    println!("w\t\t\tGreedy\t\tOptimal (brute-force)");
    // brute-force = (x_max+1).pow(m.len())
    //             = (w / m.iter().min().unwrap() + 1).pow(m.len())
    // for w=5 already:
    //             = (5 / 1 + 1).pow(9)
    //             = 6.pow(9)
    //             = 10 077 696
    for w in 0.. {
        let greedy_solution = change_greedy(w, &coins);
        let optimal_brute_force_solution = change_optimal_brute_force(w, &coins);

        print!("{}\t\t\t{}\t\t\t{}",
               w,
               greedy_solution.iter().sum::<u128>(),
               optimal_brute_force_solution.iter().sum::<u128>()
        );

        // Add a fourth and a fifth column to a row iff Greedy does not give an optimal solution:
        if greedy_solution.iter().sum::<u128>() != optimal_brute_force_solution.iter().sum::<u128>() {
            println!("\t\t\t{:?}\t\t\t{:?}", greedy_solution, optimal_brute_force_solution);
        } else {
            println!();
        }
    }
}
