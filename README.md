Accompanying material to the conference paper:
Lips, J., Lens, H. and Conti, P. (2025). “[Soft Sensor Design Using Hierarchical Multi-Fidelity Modeling with Bayesian Optimization for Input Variable Selection](https://skoge.folk.ntnu.no/prost/proceedings/dycops-2025/DYCOPS2025_offline_program/contents/papers/0074.pdf)”. 
In Proceedings of the 14th IFAC Symposium on Dynamics and Control of Process Systems, including Biosystems – DYCOPS 2025.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fc7b9ba2-fd92-4955-aa43-237be47f51ec" alt="graphical_abstract_new5" width="700" />
</p>

The repository contains code to run the first example of the paper (download the repo, run the outer loop in matlab, it will call python for the inner loop which relies on the progressive_network).
The appendix to the paper is also available here.

## Abstract
Soft sensors, or inferential sensors, are crucial in quality and process control systems because they allow for efficient, online estimation of essential quantities that are otherwise difficult or expensive to measure directly. In many applications, it is common to use cost-effective measurement equipment, offering faster data collection than high-fidelity measurements, albeit at the price of reduced accuracy.
These low-fidelity data can provide useful information to enhance the estimation of output quantities of interest, thereby facilitating the design of inferential control systems. 
In this work, we introduce an innovative approach to soft sensing by employing hierarchical, multi-fidelity surrogate models as soft sensors, integrated with Bayesian optimization for input variable selection.
Our method creates a parsimonious model by identifying and organizing relevant inputs into a fidelity hierarchy, which enables a multi-fidelity neural network to sequentially refine estimations by extracting crucial information progressively.

First, we showcase the effectiveness of the proposed framework on a numerical benchmark, then we use our method to create a surrogate model as soft sensor for accurately determining the atmospheric particulate matter concentration (PM2.5) using real data collected from low-cost sensors. 
