# Rowdy_Activation_Functions

The Rowdy activation function codes written in Tensorflow 1.14

We propose a new type of neural networks, Kronecker neural networks (KNNs), that form a generalframework  for  neural  networks  with  adaptive  activation  functions.   KNNs  employ  the  Kroneckerproduct,  which  provides  an  efficient  way  of  constructing  a  very  wide  network  while  keeping  thenumber of parameters low.  Our theoretical analysis reveals that under suitable conditions, KNNsinduce a faster decay of the loss than that by the feed-forward networks.  This is also empiricallyverified through a set of computational examples.  Furthermore, under certain technical assumptions,we establish global convergence of gradient descent for KNNs.  As a specific case, we propose theRowdyactivation function that is designed to get rid of any saturation region by injecting sinusoidalfluctuations, which include trainable parameters.  The proposed Rowdy activation function can beemployed  in  any  neural  network  architecture  like  feed-forward  neural  networks,  Recurrent  neuralnetworks, Convolutional neural networks etc.  The effectiveness of KNNs with Rowdy activation isdemonstrated through various computational experiments including function approximation usingfeed-forward neural networks, solution inference of partial differential equations using the physics-informed neural networks, and standard deep learning benchmark problems using convolutional andfully-connected neural networks.



Code: TBA 


Reference for Rowdy activation functions:

1. A.D. Jagtap, K.Kawaguchi, G.E.Karniadakis, Adaptive activation functions accelerate convergence in deep and physics-informed neural networks, Journal of Computational Physics, 404 (2020) 109136. (https://doi.org/10.1016/j.jcp.2019.109136)

       @article{jagtap2020adaptive,
       title={Adaptive activation functions accelerate convergence in deep and physics-informed neural networks},
       author={Jagtap, Ameya D and Kawaguchi, Kenji and Karniadakis, George Em},
       journal={Journal of Computational Physics},
       volume={404},
       pages={109136},
       year={2020},
       publisher={Elsevier}
       }

2. A.D.Jagtap, K.Kawaguchi, G.E.Karniadakis, Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 20200334, 2020. (http://dx.doi.org/10.1098/rspa.2020.0334).


       @article{jagtap2020locally,
       title={Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks},
       author={Jagtap, Ameya D and Kawaguchi, Kenji and Em Karniadakis, George},
       journal={Proceedings of the Royal Society A},
       volume={476},
       number={2239},
       pages={20200334},
       year={2020},
       publisher={The Royal Society}
       }


3. A.D. Jagtap, Y. Shin, K. Kawaguchi, G.E. Karniadakis, Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions, Neurocomputing, 468, 165-180, 2022. (https://www.sciencedirect.com/science/article/pii/S0925231221015162)

       @article{jagtap2022deep,
       title={Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions},
       author={Jagtap, Ameya D and Shin, Yeonjong and Kawaguchi, Kenji and Karniadakis, George Em},
       journal={Neurocomputing},
       volume={468},
       pages={165--180},
       year={2022}
       }



