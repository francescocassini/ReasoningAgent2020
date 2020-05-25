# BabyAI and Restraining Bolts
Our aim is to train an agent over LDLf formulae with Restraining Bolt, a device which takes inspiration from the movie "Star Wars", that is capable of limiting the actions of the agent on which it is installed.

## Problem description
Our work is based on the idea showed in the paper *BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning*. [1] 
In this paper, the agent is requested to learn a combinatorial natural language and to act accordingly to accomplish some requested tasks.
  
The initial attempt was to try to implement the Restraining Bolt directly on BabyAI but we soon discovered that it is not a trivial task, so we decided to mantain the same concepts of BabyAI but to present a simplified version of it in order to prove that what we implemented can be also realized for BabyAI in future works.

Typical requests of BabyAI environment are of the type:
* Open the door
* Get the box
* Move the box
* ...

As we can notice, these are all atomic requests that can be performed in a relative easy way after some training. The problem of BabyAI arises when these requests are combined together to form a more complex sentence.

In fact, this is exactly what BabyAI does and it is also what we do in our project, with good results too. 

But it is important to understand that, while BabyAI aims to teach a combinatorial language to its agent, our final result is to teach to our agent how to perform a specific task, which is slightly different. Despite of it, we still believe that our implementation is a good starting point for other works.

## NLP

A module responsible of the Natural Language Processing is needed in order to acquire spoken sentences and translate them in LDLf formulae. 

For that reason, we wrote *nl2ldl.py* which returns an LDLf formula in the form of a regular expression. The script is based on a combination of other existing methods which are believed as being part of thhe state of the art. [2]

## Environment

The choice of the environment takes in account the similarity with the BabyAI one, especially with regards to the base logic. That's why we focused on the already implemented games provided by the repository of the paper *Imitation Learning over Heterogeneous Agents with Restraining Bolts*. [3]

In that repository we can find three games:

* Breakout
* Sapientino
* Minecraft

Among these three, Minecraft has resulted to be the most suitable environment for our purpose. Sapientino-like environments are not good because a logic based only on colours is not sufficient in this case.

In its original version, Minecraft is presented with a predefined set of actions and objects. We added another action to the existing ones which is called *Move*, in order to manage *pick-and-drop* situations. Then we also replaced the entire set of objects accordingly to the ones that are in BabyAI, keeping just the classes *Resources* and *Tools*, and adding a new class called *Obstacles*. 
This class is responsible of creating obstacles on the scene which avoid the agent to walk on certain cells of the grid. 

Finally, the set of actions and objects as they are in the code, are defined as follows:

* **Actions**
  *  Get
  *  Use
  *  Move

* **Objects**
  *  *Resources*:
      *    Door
      *    Ball
      *    Base
      *    Box
  *  *Tools*:
      *    Key
  *  *Obstacles*:
      *    Wall
    
## Game description

The rules of our game are very easy to understand. We have an agent which must accomplish the tasks described by the user, firstly provided in Natural Language and then translated under the form of LDLf formulae. 

The environment is characterized by four rooms that - in the most complex scenario - are all blocked by some doors and walls.

At the beginning of any level, the doors are always locked.

The agent must unlock the door first to access the other room. It can do so by using the key that is put in each room. 

Just for giving you a better understanding about the environment, in the repository you can find an example in which the agent must satisfy the goal:

**Use the key and get the box**

This is our *Level 1* and it is possible to notice that the box is placed in the second room, beyond the locked door. 
So the agent is requested to unlock the door before proceeding and getting the box.

## References
[1] https://arxiv.org/abs/1810.08272

[2] https://drops.dagstuhl.de/opus/volltexte/2019/11375/pdf/LIPIcs-TIME-2019-17.pdf

[3] https://www.dis.uniroma1.it/~degiacom/papers/2020/icaps2020dfip.pdf
