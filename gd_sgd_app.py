import streamlit as st
import numpy as np
import plotly.graph_objs as go

def convex_function(x, y):
    return x**2 + y**2

def non_convex_function(x, y):
    return np.sin(x) * np.cos(y)

def gradient_descent(func, grad_func, start, learning_rate, n_iter):
    path = [start]
    for _ in range(n_iter):
        grad = grad_func(path[-1])
        next_point = path[-1] - learning_rate * grad
        path.append(next_point)
    return np.array(path)

def stochastic_gradient_descent(func, grad_func, start, learning_rate, n_iter):
    path = [start]
    for _ in range(n_iter):
        grad = grad_func(path[-1]) + np.random.normal(0, 0.1, 2)
        next_point = path[-1] - learning_rate * grad
        path.append(next_point)
    return np.array(path)

def grad_convex(point):
    x, y = point
    return np.array([2*x, 2*y])

def grad_non_convex(point):
    x, y = point
    return np.array([np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y)])

def plot_3d_surface(func, path, title):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, opacity=0.7)])
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=func(path[:, 0], path[:, 1]), 
                               mode='lines+markers', line=dict(color='orange'), 
                               marker=dict(size=4, color='orange'), name='Path'))
    fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[func(path[0, 0], path[0, 1])],
                               mode='markers', marker=dict(size=6, color='green'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[path[-1, 0]], y=[path[-1, 1]], z=[func(path[-1, 0], path[-1, 1])],
                               mode='markers', marker=dict(size=6, color='red'), name='End'))
    fig.update_layout(title=title, scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'))
    return fig

st.title("Convex and Non-Convex Optimization Problems")

tab1, tab2 = st.tabs(["Gradient Descent", "Stochastic Gradient Descent"])

st.sidebar.header("Parameters")

learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
n_iter = st.sidebar.slider("Number of Iterations", 10, 100, 50)
convex_start_x = st.sidebar.slider("Convex Start X", -3.0, 3.0, 2.5)
convex_start_y = st.sidebar.slider("Convex Start Y", -3.0, 3.0, 2.5)
non_convex_start_x = st.sidebar.slider("Non-Convex Start X", -3.0, 3.0, 2.5)
non_convex_start_y = st.sidebar.slider("Non-Convex Start Y", -3.0, 3.0, 2.5)

convex_start = np.array([convex_start_x, convex_start_y])
non_convex_start = np.array([non_convex_start_x, non_convex_start_y])

st.sidebar.markdown("""
### Author and Further Reading
[Author: Roman Paolucci](https://romanmichaelpaolucci.github.io).
""")

st.sidebar.markdown("""
### Role of Optimization in Deep Learning and AI
Minimization is the backbone of deep learning and artificial intelligence. Most of the time, we have a specific goal (even if it is hard to define), and our model makes predictions toward that goal using some weights. We want these weights to get better and better at achieving the goal. This process involves defining a loss function that measures the discrepancy between the model's predictions and the actual outcomes. By minimizing this loss function, we can optimize the model's weights to improve its performance. In the following interactive app x and y act as our "weights" and z acts as our loss function. In other words, we want to find the best x and y to create the lowest z.
""")

st.sidebar.markdown("""
### Why Consider Non-Convex Optimization Problems?
Non-convex optimization problems are common in deep learning due to the high-dimensional and complex nature of neural network architectures. The loss functions of neural networks often have many local minima, saddle points, and complex landscapes. Stochastic gradient descent (SGD) helps in navigating these landscapes by providing a method that can escape local minima and potentially find better solutions by leveraging its stochastic nature. Understanding and visualizing these optimization paths are crucial for developing effective deep learning models.
""")

with tab1:
    st.header("Gradient Descent")
    st.write("Visualizing gradient descent on convex and non-convex functions.")

    with st.expander("Gradient Descent Algorithm and Math"):
        st.markdown(r"""
        ### Gradient Descent Algorithm
        **Step-by-step Algorithm**:
        1. Initialize starting point $\mathbf{x}_0$.
        2. For each iteration $t$:
           - Compute the gradient $\nabla f(\mathbf{x}_t)$.
           - Update the current point: $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t)$.

        **Mathematical Formulation**:
        $$
        \mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t)
        $$
        where:
        - $\mathbf{x}_t$ is the current point.
        - $\alpha$ is the learning rate.
        - $\nabla f(\mathbf{x}_t)$ is the gradient of the function at $\mathbf{x}_t$.
        """)

    convex_path_gd = gradient_descent(convex_function, grad_convex, convex_start, learning_rate, n_iter)
    non_convex_path_gd = gradient_descent(non_convex_function, grad_non_convex, non_convex_start, learning_rate, n_iter)

    st.plotly_chart(plot_3d_surface(convex_function, convex_path_gd, "Convex Function (GD)"))
    st.plotly_chart(plot_3d_surface(non_convex_function, non_convex_path_gd, "Non-Convex Function (GD)"))

with tab2:
    st.header("Stochastic Gradient Descent")
    st.write("Visualizing stochastic gradient descent on convex and non-convex functions.")

    with st.expander("Stochastic Gradient Descent Algorithm and Math"):
        st.markdown(r"""
        ### Stochastic Gradient Descent Algorithm
        **Step-by-step Algorithm**:
        1. Initialize starting point $\mathbf{x}_0$.
        2. For each iteration $t$:
           - Compute a stochastic approximation of the gradient $\nabla f(\mathbf{x}_t) + \text{noise}$.
           - Update the current point: $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \left(\nabla f(\mathbf{x}_t) + \text{noise}\right)$.

        **Mathematical Formulation**:
        $$
        \mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \left(\nabla f(\mathbf{x}_t) + \text{noise}\right)
        $$
        where:
        - $\mathbf{x}_t$ is the current point.
        - $\alpha$ is the learning rate.
        - $\nabla f(\mathbf{x}_t)$ is the gradient of the function at $\mathbf{x}_t$.
        - $\text{noise}$ is a small random perturbation.
        """)

    convex_path_sgd = stochastic_gradient_descent(convex_function, grad_convex, convex_start, learning_rate, n_iter)
    non_convex_path_sgd = stochastic_gradient_descent(non_convex_function, grad_non_convex, non_convex_start, learning_rate, n_iter)

    st.plotly_chart(plot_3d_surface(convex_function, convex_path_sgd, "Convex Function (SGD)"))
    st.plotly_chart(plot_3d_surface(non_convex_function, non_convex_path_sgd, "Non-Convex Function (SGD)"))
