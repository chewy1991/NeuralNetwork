using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace NeuralNetwork.Layers
{
    public class Activation_RELU : ILayer
    {
        public Activation_RELU(){}

        public NDarray Forward(NDarray inputs)
        {
            return np.maximum(np.array(0), inputs);
        }
    }
}
