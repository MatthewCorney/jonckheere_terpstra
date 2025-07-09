if (!requireNamespace("PMCMRplus", quietly = TRUE)) install.packages("PMCMRplus")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

library(PMCMRplus)
library(jsonlite)

set.seed(42)
number_test_cases=2

alternatives <- c("two.sided", "greater", "less")
group_sizes <- list(5,10,20,30)
group_numbers <- list(3,4,5)
continuity_options <- list(TRUE, FALSE)

results <- list()
count=1
for (i in 1:number_test_cases) {
  for (alt in alternatives){
    for (continuity in continuity_options){
      for (group_size in group_sizes){
        for (group_number in group_numbers){
          
          
          g <- rep(1:group_number, each = group_size)
          slope <- runif(1, -1, 1)
          x <- rnorm(length(g)) + slope * g
          
          test_result <- jonckheereTest(x, g, alternative = alt, continuity=continuity)
          
          if (alt == 'less') {
            alt_p <- 'decreasing'
          } else if (alt == 'greater') {
            alt_p <- 'increasing'
          } else if (alt == 'two.sided') {
            alt_p <- 'two_sided'
          }
          results[[count]] <- list(
            ties=any(duplicated(x)),
            more_than_100_obs=100<length(x),
            x=x,
            g=g,
            slope=slope,
            id = i,
            continuity=continuity,
            nperm=NULL,
            group_size = group_size,
            group_number = group_number,
            alt = alt_p,
            statistic = unname(test_result$estimates),
            zstat = unname(test_result$statistic),
            p_value = unname(test_result$p.value),
            significant = 0.05 > unname(test_result$p.value),
            
          )
          count <- count+ 1
        }
      }
    }
  }
}

write_json(results, "jonckheere_test_results_PMCMRplus.json", pretty = TRUE, auto_unbox = TRUE)