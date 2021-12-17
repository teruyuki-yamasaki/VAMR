# From images to localization



## Familiarize ourselves with RANSAC

[test_ransacPolynomial.py](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise07/code/test_ransacPolynomial.py)
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise07/results/ransacdeg1.png"/>
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise07/results/ransacdeg2.png"/>
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise07/results/ransacdeg3.png"/>

RANSAC [(wiki)](https://en.wikipedia.org/wiki/Random_sample_consensus)
```
Given:
    data – A set of observations.
    model – A model to explain observed data points.
    n – Minimum number of data points required to estimate model parameters.
    k – Maximum number of iterations allowed in the algorithm.
    t – Threshold value to determine data points that are fit well by model.
    d – Number of close data points required to assert that a model fits well to data.

Return:
    bestFit – model parameters which best fit the data (or null if no good model is found)

iterations = 0
bestFit = null
bestErr = something really large

while iterations < k do
    maybeInliers := n randomly selected values from data
    maybeModel := model parameters fitted to maybeInliers
    alsoInliers := empty set
    for every point in data not in maybeInliers do
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
        end if
    end for
    if the number of elements in alsoInliers is > d then
        // This implies that we may have found a good model
        // now test how good it is.
        betterModel := model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr := a measure of how well betterModel fits these points
        if thisErr < bestErr then
            bestFit := betterModel
            bestErr := thisErr
        end if
    end if
    increment iterations
end while

return bestFit
```
