using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace NeuralNetwork.Layers
{
    public interface ILayer
    {
        NDarray Forward(NDarray inputs);
    }
}
