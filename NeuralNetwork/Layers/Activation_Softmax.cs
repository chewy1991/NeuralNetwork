using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace NeuralNetwork.Layers
{
    public class Activation_Softmax : ILayer
    {
        public Activation_Softmax() { }

        public NDarray Forward(NDarray inputs)
        {
            var exp_values = np.exp(inputs - np.max(inputs, new int[] {1}, null, true));

            var probabilities = exp_values / np.sum(exp_values, 1, null, null, true);

            return probabilities;
        }
    }
}
