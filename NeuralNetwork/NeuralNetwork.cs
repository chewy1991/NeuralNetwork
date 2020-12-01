using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using NeuralNetwork.Layers;
using NeuralNetwork.LossClasses;
using Numpy;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private List<NDarray> class_targets = new List<NDarray>()
                                              {
                                                  np.array(new double[] {1, 0 ,0, 0})
                                                , np.array(new double[] {0, 1, 0, 0})
                                                , np.array(new double[] {0, 1, 0, 0})
                                                , np.array(new double[] {0, 0, 1, 0})
                                              };

        public void HiddenLayer1()
        {
            var input = CreateRandomInputData.CreateInputData(100, 3); 
            
            var shape = input.shape.Dimensions;
            var dense_layer1 = new Dense_Layer(shape[1], shape[0]);
            var outdense = dense_layer1.Forward(input);

            var activation = new Activation_RELU();
            var outactivation = activation.Forward(outdense);

            shape = outactivation.shape.Dimensions;
            var dense_layer2 = new Dense_Layer(shape[1], shape[0]);
            var outdense2 = dense_layer2.Forward(outactivation);

            var softmax = new Activation_Softmax();
            var soft = softmax.Forward(outdense2);
            var softshape = soft.shape.Dimensions;
            var classtargets = np.maximum( np.array(0),np.random.randn(softshape[1], softshape[0]));

            var loss = new CalculateEntropyLoss();
            var entropyloss = loss.CalcEntropyLoss(soft, classtargets);

            
            Console.WriteLine(entropyloss);
            Console.ReadLine();
        }

        public void OutputLayer()
        {

        }
    }
}
