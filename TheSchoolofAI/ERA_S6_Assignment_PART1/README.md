# Details about the simulation done in excel to demonstrate the back propagation
## Details about the model network --> BLOCK 1
1. Model has 2 inputs i1 and i2
2. First Hidden layers has 2 neurons with values h1 and h2. First fully connected layer has 4 weights : w1 (0.15), w2(0.2), w3(0.25), w4(0.3)
3. SIGMOID function has been used to add non-linearity. This produces outputs from the first FC layer as a_h1 and a_h2
4. 4 weights w5(0.4), w6(0.45), w7(0.5) and w8(0.55) are used to produce the output o1 and o2
5. SIGMOID function has been used to add non-linearity. This produces final outputs from the output layer as a_o1 and a_o2 
## Diagram is depicts that outputs a_o1 and a_o2 are compared with targets t1 and t2 respectively. This helps to calculate the total loss E_total = E1 + E2
## BLOCK 2
1. Depicts the derivatives calculation to calculate the derivative w5 w.r.t. total loss. However w5 is used to calculate the a_o1 and hence E1.
2. Hence ∂E_total/∂w5 = ∂(E1 + E2)/∂w5 corresponds to ∂E1/∂w5
3. Derivative chain rule is applied to break this down to individual components such as ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5
## BLOCK 3 
1. Depicts the calculation of derivative of ∂E_total against the weight in output layer i.e. w5, w6, w7 and w8
## BLOCK 4 
1. Depicts the calculation of derivative of ∂E_total/∂a_h1 and ∂E_total/∂a_h2
2. Derivative chain rule is applied to break this down to individual components such as ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
## BLOCK 5
1. Derivative of ∂E_total against weights w1, w2 and w3 is calculated
## BLOCK 6
1. Depicts the calculation of derivative of ∂E_total against each of the weight in hidden layer i.e. w1, w2, w3 and w4
## Tabular calculations have been added to calcuate the loss ∂E_total based on these formulas
## Simulation performed for different learning rates to indicate the model training performance. This is depicted in the diagram below
![Model Performance at different learning rates]([image link](https://github.com/prasad0679/Prasad_ERA_Repo/blob/master/TheSchoolofAI/ERA_S6_Assignment_PART1/Losses_atDiff_LearningRates.JPG))
