#include<iostream>
#include<algorithm>
#include<list>
#include<math.h>

#include"net.h"

#include<QDebug>
#include<QTime>
#include<QProcess>
#include<QFile>
#include<QRandomGenerator>

double Noise(int low, int high)
{
    return (double)((qrand() % ((high + 1) - low) + low)/300.0);
}

using namespace std;

int main()
{
    float step = 0.001;                                 //Шаг выборки, графика
    QFile learndata("learndata.txt");                   //Файл с обучающей выборкой
    QFile file("data.txt");                             //Файл с ответом нейронной сети
    QTextStream writeStream(&learndata);                //Сейчас будем пистаь в файл leandata.txt

    std::vector<std::pair<double, double>> datalearn;   //Обучающая выборка

    Net *mynet = new Net();                             //Создаем нейросеть. Конфигурация сети описывается в конструкторе.

    if (learndata.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        for (double x = 0; x < 4*M_PI; x +=step)
        {
            double noise = Noise(-100,100);
            datalearn.push_back(make_pair(x, (noise + sin(x)+1)/2));                                 //Формируем обучабщию выборку с шумом
            writeStream << x <<", " << (sin(x)+1)/2 << ", " << noise + (sin(x)+1)/2 << ", " << '\n'; //Паралельно пиши в файл для gnuplot
        }

        learndata.close();  // Закрываем файл и пишем в него.
    }


    int end = datalearn.size();

    //Начало обучения
    for (int qq = 0; qq < 10000000; qq++)
    {
        int id = qrand() % end; // Генерируем любой индекс для обучающей выборки.
        double arrInput[] = {datalearn[id].first}; // Формируем входной вектор  по рандомному id из выборки.
        double arrTarget[] = {datalearn[id].second}; // формируем ветор цель.
        mynet->train(arrInput,  arrTarget); // Тренеруем
    }



    if (file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream writeStream(&file); // Теперь записываем ответы нейронки на запросы

        double arrQuery[] = {0};
        for (double x = 0; x < 4*M_PI; x += step)
        {
            arrQuery[0] = x;
            double* ans = mynet->query(arrQuery); // Получаем ответ нейронки
            double err = (sin(x)+1)/2 - ans[0];   // Считаем ошибку, для сравнения
            std::cout << x <<", " << (sin(x)+1)/2 << ", " << ans[0] << ", " << err << ", " << '\n';
            writeStream << x <<", " << (sin(x)+1)/2 << ", " << ans[0] << ", "<< err << ", " << '\n'; //Пишим в файл data.txt
        }
        file.close();
    }

    //Запускаем gnuplot. ./
    QProcess *proc =  new QProcess();
    QProcess *proc2 =  new QProcess();
    proc->setWorkingDirectory(".");
    proc2->setWorkingDirectory(".");
    proc->start("./plotdata.gpi");
    proc2->start("./datalearn.gpi");
    proc->waitForFinished();
    proc2->waitForFinished();
    delete proc;
    delete proc2;

    return 0;
}
