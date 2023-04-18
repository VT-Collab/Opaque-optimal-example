# Should Collaborative Robots be Transparent

Code for the example in Section 4.C of our paper ["Should Collaborative Robots be Transparent?"](link).

## Instructions

Run the code using `python main.py`
 - To see optimal behavior that is *fully opaque*, include the argument '--example fully'
 - To see optimal behavior that is *rationally opaque* but not *fully opaque*, use the argument '--example rationally'

## Results

 - *--example fully* | the human's final belief should be 0.0 for each type of robot and each type of human. In other words, the system is fully opaque and the robot's optimal behavior convinces the human that the robot is confused.
 - *--example rationally* | the human's final belief should be 0.0 for capable and confused robots if the human is rational. When the human is *irrational*, the final belief for confused is 0.0 and the final belief for capable is 0.4. By perturbing the system the irrational human uncovers information about the robot's type.