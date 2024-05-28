#include "net.h"
#include <iostream>
#include <math.h>
#include <vector>

Net::Net()
{
    //Инициализация нейронки

    intputNeurons = 1;
    outputNeurons = 1;
    nlCount = 4;


    list = new std::vector<Layers>(nlCount);

    list->at(0).initLayer(intputNeurons, 20);
    list->at(1).initLayer(20, 20);
    list->at(2).initLayer(20, 20);
    list->at(nlCount - 1).initLayer(20, outputNeurons);

}

Net::~Net()
{
    for (int i = 0; i < nlCount; i++)
    {
        delete [] list->data()[i].hidden;
        list->data()[i].hidden = nullptr;

        delete [] list->data()[i].errors;
        list->data()[i].errors = nullptr;

        for (int inp = 0; inp < list->data()[i].in; inp++)
        {
            delete [] list->data()[i].matrix[inp];
            list->data()[i].matrix[inp] = nullptr;
        }

        delete [] list->data()[i].matrix;
        list->data()[i].matrix = nullptr;
    }
}

void Net::PrintArray(double *arr, int s)
{
    std::cout << "--------------" << std::endl;
    for (int inp = 0; inp < s; inp++)
    {
        std::cout << arr[inp] << std::endl;
    }
}

double* Net::query(double *in)
{
    inputs = in;
    RunNet(false);
    return list->at(nlCount - 1).hidden;
}

void Net::train(double *in, double *targ)
{
    if (in) inputs = in;
    if (targ) targets = targ;

    RunNet(true);
}

void Net::backPropagation()
{
    list->at(nlCount - 1).caltOutError(targets);

    for (int i = nlCount - 2; i >= 0; i--)
    {
        list->at(i).calcHiddnError(list->at(i+1).getError(), list->at(i+1).getMatrix(), list->at(i+1).getInCount(), list->at(i+1).getOutCount());
    }

    for (int i = nlCount - 1; i > 0; i--)
    {
        list->at(i).updateMatrix(list->at(i-1).getHidden());
    }

    list->at(0).updateMatrix(inputs);
}

void Net::RunNet(bool ok)
{
    list->at(0).runtHidden(inputs);

    for (int i = 1; i < nlCount; i++)
    {
        list->at(i).runtHidden(list->at(i-1).getHidden());
    }

    if (!ok)
    {
        //std::cout << "RunNet" << std::endl;
        for (int out = 0; out < outputNeurons; out++)
        {
           //std::cout << list->at(nlCount - 1).hidden[out] << std::endl;
        }
        return;
    } else {

        backPropagation();
    }


}

/////////////////////////////////////////   LAYERS   //////////////////////////////
double Net::Layers::sigmoida(double val)
{
    return (1.0 / (1.0 + exp(-val)));
}

double Net::Layers::sigmoidaDerivate(double val)
{
    return (val * (1.0 - val));
}

int Net::Layers::getInCount()
{
    return in;
}

int Net::Layers::getOutCount()
{
    return out;
}

double *Net::Layers::getHidden()
{
    return hidden;
}

double *Net::Layers::getError()
{
    return errors;
}

double **Net::Layers::getMatrix()
{
    return matrix;
}

void Net::Layers::updateMatrix(double* enteredVal)
{
    for (int ou = 0; ou < out; ou++)
    {
        for (int hid = 0; hid < in; hid++)
        {
            matrix[hid][ou] += (learnRate * errors[ou] * enteredVal[hid]);
        }
        matrix[in][ou] += (learnRate * errors[ou]);
    }
}

void Net::Layers::initLayer(int inputs, int outputs)
{
    //--- Выделение памяти и инициализация слоя.
    in=inputs;
    out=outputs;
    errors = (double*) malloc((out)*sizeof(double));
    hidden = (double*) malloc((out)*sizeof(double));

    matrix = (double**) malloc((in+1)*sizeof(double*));
    for(int inp =0; inp < in+1; inp++)
    {
        matrix[inp] = (double*) malloc(out*sizeof(double));
    }
    for(int inp =0; inp < in+1; inp++)
    {
        for(int outp =0; outp < out; outp++)
        {
            matrix[inp][outp] =  randWeight ; // randWeight;
        }
    }
}

void Net::Layers::runtHidden(double *inputs)
{
    for (int hid = 0; hid < out; hid++)
    {
        double temp = 0.0;
        for (int inp = 0; inp < in; inp++)
        {
            temp += inputs[inp] * matrix[inp][hid];
        }
        temp += matrix[in][hid]; // Порог

        hidden[hid] = sigmoida(temp);
    }
} // Прогон скрытого слоя.




void Net::Layers::calcHiddnError(double *targets, double **outWeights, int inS, int outS)
{
    for (int hid = 0; hid < inS; hid++)
    {
        errors[hid] = 0.0;
        for (int ou = 0; ou < outS; ou++)
        {
            errors[hid] += targets[ou] * outWeights[hid][ou];
        }
        errors[hid] *= sigmoidaDerivate(hidden[hid]);
    }
}

void Net::Layers::caltOutError(double *targets)
{
    for (int ou = 0; ou < out; ou++)
    {
        errors[ou] = (targets[ou] - hidden[ou]) * sigmoidaDerivate(hidden[ou]);
    }
}


























