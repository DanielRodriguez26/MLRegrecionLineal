from turtle import color
import numpy as np
import matplotlib.pyplot as plt


def estimated_coef(x, y):
    # numeros de observaciones
    n = np.size(x)

    # media del vector de  X y Y
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calcular la desviación cruzada y la desviación sobre x
    SS_xy = np.sum(x*y) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # cálculo de coeficientes de regresión
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # trazar los puntos reales como diagrama de dispersión
    plt.scatter(x, y, color='red', marker="o", s=30)

    # prediccion de la respuesta del vector
    y_pred = b[0] + b[1] * x

    # grafica de la regrecion lineal
    plt.plot(x, y_pred, color='g')

    plt.xlabel('x')
    plt.ylabel('y')

    #mostrar las graficas
    plt.show()

def main():
    # visualizacion de los datos de prueba
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimacion de coeficiente
    b = estimated_coef(x, y)
    print(f"Estimated coefficients:\
            \n b_0 = {b[0]}  \
            \nb_1 = {b[1]}")

    # Grafica de la regrecion lineal
    plot_regression_line(x, y, b)


if __name__ == '__main__':
    main()
