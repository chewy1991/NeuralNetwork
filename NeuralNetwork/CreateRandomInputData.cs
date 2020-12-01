using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Numpy;


namespace NeuralNetwork
{
    public static class CreateRandomInputData
    {
        public static NDarray CreateInputData(int inputs, int neurons)
        {
            List<NDarray> input = new List<NDarray>();
            

            for (int i = 0; i < neurons; i++)
            {
                var inputArray = new double[inputs];

                for (int j = 0; j < inputs; j++)
                {
                    Random rand = new Random();
                    int multiplier = rand.Next(-10, 10);
                    while (multiplier == 0)
                    {
                        multiplier = rand.Next(-1, 1);
                    }

                    multiplier = multiplier > 0 ? 1 : -1;

                    double val = rand.NextDouble() * multiplier;
                    inputArray[j] = val;
                }
                input.Add(np.array(inputArray));
            }

            return np.array(input);
        }
    }
}
