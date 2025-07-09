if (!requireNamespace("clinfun", quietly = TRUE)) install.packages("clinfun")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

library(clinfun)
library(jsonlite)

set.seed(42)
number_test_cases=2

alternatives <- c("two.sided", "increasing", "decreasing")
nperm_options <- list(1000, 5000)
group_sizes <- list(5,10,20,30)
group_numbers <- list(3,4,5)
results <- list()
count=1
for (i in 1:2) {
  for (alt in alternatives){
    for (nperm in nperm_options){
      for (group_size in group_sizes){
        for (group_number in group_numbers){
          
          g <- rep(1:group_number, each = group_size)
          slope <- runif(1, -1, 1)
          x <- rnorm(length(g)) + slope * g
          
          test_result <- jonckheere.test(x, g, alternative = alt, nperm = nperm)

          if (alt == 'decreasing') {
            alt_p <- 'decreasing'
          } else if (alt == 'increasing') {
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
            
            continuity=NULL,
            nperm=nperm,
            
            group_size = group_size,
            group_number = group_number,
            
            alt = alt_p,
            
            statistic = unname(test_result$statistic),
            zstat=NULL,
            p_value = unname(test_result$p.value),
            significant = 0.05 > unname(test_result$p.value)
          )
          count <- count+ 1
          
          }
        }
      }
    }
  }

write_json(results, "jonckheere_test_results_clinfun.json", pretty = TRUE, auto_unbox = TRUE)
