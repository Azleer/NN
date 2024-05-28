#ifndef NET_H
#define NET_H
#include <math.h>
#include<iostream>
#include<vector>
#include<QRandomGenerator>


#define learnRate 0.13
#define randWeight (( ((double)rand() / (double)RAND_MAX) - 0.5)* pow(out,-0.5))

class Net
{
public:
    Net();
    ~Net();

    struct Layers
    {
        int in; //"выход с предыдущего слоя"
        int out; //"выход this слоя"
        double** matrix; //Матрица весов для данного слоя;
        double* hidden; // Массив скрытых нейронов
        double* errors; // Текущая ошибка в алгоритме обратного распрастронении ошибки.
        double sigmoida(double val); // Функция активации
        double sigmoidaDerivate(double val); // Производная функции активации


        int getInCount();       //
        int getOutCount();      //
        double* getHidden();     // Просто getters
        double* getError();      //
        double **getMatrix();    //
        void updateMatrix(double* enteredVal);    // Обновление весов с учетом ошибок.
        void initLayer (int intputs, int outputs); // "Конструктор" слоя + инициализация"
        void runtHidden(double *inputs); // Прогон слоя


        void calcHiddnError(double *targets, double **outWeights, int inS, int outS);
        void caltOutError(double* targets);

    };

    void PrintArray(double *arr, int s);
    double* query(double *in);
    void train(double* in, double* targ);
    void backPropagation();
    void RunNet(bool ok);

private:
    std::vector<Layers> *list = nullptr;
    int intputNeurons; // Кол-во нейронов в входном слое
    int outputNeurons; // Кол-во нейронов в выходном слое
    int nlCount;       // Количество слоев, вкл входной и выходной.

    double *inputs = nullptr;
    double *targets = nullptr;
};

#endif // NET_H
