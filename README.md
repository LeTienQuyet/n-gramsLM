# N-gram Language Model 
## Theory
* Given a sequence consisting of `i-1` words: $$w_1,w_2,\ldots,w_{i-1}$$
* We aim to determine the most probable next word, which satisfies: $$w_{i} = \arg\max_{w_{i}} P(w_i|w_1, w_2,\ldots,w_{i-1})$$ 
=> Calculating this directly for long sequences is impractical due to the exponential growth of possible word combinations.
* To simplify this, we apply the *Markov Assumption*, which states that the probability of a word depends only on the last `n-1` words, rather than the entire preceding sequence: $$P(w_j | w_1,w_2,\ldots,w_{j-1}) \approx P(w_j| w_{j-n+1},\ldots,w_{j-1})$$
=> `n-gram` language model.
* The probability of a word appearing given its preceding context is estimated using frequency counts from a corpus: $$P(w_j| w_{j-n+1},\ldots,w_{j-1})=\frac{Count(w_{j-n+1}, \ldots, w_{j-1}, w_j)}{Count(w_{j-n+1}, \ldots, w_{j-1})}$$