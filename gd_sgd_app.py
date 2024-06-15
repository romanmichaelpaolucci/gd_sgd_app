import streamlit as st
import numpy as np
import plotly.graph_objs as go

def convex_function(x, y):
    return x**2 + y**2

def non_convex_function(x, y):
    return np.sin(x) * np.cos(y) * x * y

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
    return np.array([np.cos(x) * np.cos(y) * y + np.sin(x) * np.sin(y) * x, np.cos(x) * np.cos(y) * x - np.sin(x) * np.sin(y) * y])

def simulated_annealing(func, start, temp, cooling_rate, n_iter):
    path = [start]
    current_point = start
    lowest_point = current_point
    for i in range(n_iter):
        next_point = current_point + np.random.normal(0, 1, 2)
        delta_E = func(next_point[0], next_point[1]) - func(current_point[0], current_point[1])
        if delta_E < 0 or np.exp(-delta_E / temp) > np.random.rand():
            current_point = next_point
        if func(current_point[0], current_point[1]) < func(lowest_point[0], lowest_point[1]):
            lowest_point = current_point
        path.append(current_point)
        temp *= cooling_rate
    return np.array(path), lowest_point

def plot_3d_surface(func, path, title, alphas=None, lowest_point=None):
    x_min, x_max = min(path[:, 0].min(), -6), max(path[:, 0].max(), 6)
    y_min, y_max = min(path[:, 1].min(), -6), max(path[:, 1].max(), 6)
    
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, opacity=0.7)])
    if alphas is None:
        alphas = [1.0] * len(path)
    
    for i in range(len(path) - 1):
        fig.add_trace(go.Scatter3d(
            x=path[i:i+2, 0],
            y=path[i:i+2, 1],
            z=func(path[i:i+2, 0], path[i:i+2, 1]),
            mode='lines',
            line=dict(color='orange', width=4),
            opacity=alphas[i],
            showlegend=False
        ))
    fig.add_trace(go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=func(path[:, 0], path[:, 1]),
        mode='markers',
        marker=dict(size=4, color='orange', opacity=alphas[-1]),
        name='Path'
    ))
    fig.add_trace(go.Scatter3d(
        x=[path[0, 0]],
        y=[path[0, 1]],
        z=[func(path[0, 0], path[0, 1])],
        mode='markers',
        marker=dict(size=6, color='green', opacity=alphas[0]),
        name='Start'
    ))
    
    if lowest_point is not None:
        fig.add_trace(go.Scatter3d(
            x=[lowest_point[0]],
            y=[lowest_point[1]],
            z=[func(lowest_point[0], lowest_point[1])],
            mode='markers',
            marker=dict(size=6, color='red', opacity=alphas[-1]),
            name='Lowest Observed'
        ))
    
    fig.update_layout(title=title, scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'))
    return fig

st.title("Convex and Non-Convex Optimization Problems")

st.sidebar.markdown("""
### Author and Further Reading
[Author: Roman Paolucci](https://romanmichaelpaolucci.github.io).
""")

st.sidebar.image("rmp_profile.png", use_column_width=True)


st.sidebar.markdown("""
### Role of Optimization in Deep Learning and AI
Optimization is key in deep learning and AI, where models aim to minimize a loss function to improve performance. Parameters (x, y) are adjusted to find the optimal values that result in the lowest loss (z).
""")

st.sidebar.markdown("""
### Importance of Non-Convex Optimization
Deep learning often deals with non-convex problems due to complex neural network architectures. These problems have many local minima and saddle points, making optimization challenging. Stochastic methods like SGD help navigate these landscapes, potentially finding better solutions. Visualizing these paths is crucial for understanding and improving model performance.
""")

tab1, tab2, tab3 = st.tabs(["Gradient Descent", "Stochastic Gradient Descent", "Simulated Annealing"])

st.sidebar.header("Parameters")

learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
n_iter = st.sidebar.slider("Number of Iterations", 10, 100, 50)
convex_start_x = st.sidebar.slider("Convex Start X", -3.0, 3.0, 2.5)
convex_start_y = st.sidebar.slider("Convex Start Y", -3.0, 3.0, 2.5)
non_convex_start_x = st.sidebar.slider("Non-Convex Start X", -3.0, 3.0, 2.5)
non_convex_start_y = st.sidebar.slider("Non-Convex Start Y", -3.0, 3.0, 2.5)
temp = st.sidebar.slider("Initial Temperature (Simulated Annealing)", 1.0, 10.0, 5.0)
cooling_rate = st.sidebar.slider("Cooling Rate (Simulated Annealing)", 0.8, 0.99, 0.95)

convex_start = np.array([convex_start_x, convex_start_y])
non_convex_start = np.array([non_convex_start_x, non_convex_start_y])

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

with tab3:
    st.header("Simulated Annealing")
    st.write("Visualizing simulated annealing on a non-convex function.")

    with st.expander("Simulated Annealing Algorithm and Math"):
        st.markdown(r"""
        ### Simulated Annealing Algorithm
        **Step-by-step Algorithm**:
        1. Initialize starting point $\mathbf{x}_0$ and temperature $T$.
        2. For each iteration $t$:
           - Generate a new point $\mathbf{x}'$ in the neighborhood of the current point $\mathbf{x}_t$.
           - Compute the change in function value $\Delta E = f(\mathbf{x}') - f(\mathbf{x}_t)$.
           - If $\Delta E < 0$, accept the new point $\mathbf{x}_{t+1} = \mathbf{x}'$.
           - If $\Delta E \geq 0$, accept the new point with a probability $\exp\left(\frac{-\Delta E}{T}\right)$.
           - Update the temperature $T$.

        **Mathematical Formulation**:
        $$
        \mathbf{x}_{t+1} =
        \begin{cases} 
        \mathbf{x}' & \text{if } \Delta E < 0 \\
        \mathbf{x}' & \text{with probability } \exp\left(\frac{-\Delta E}{T}\right) \text{ if } \Delta E \geq 0 \\
        \mathbf{x}_t & \text{otherwise}
        \end{cases}
        $$
        where:
        - $\mathbf{x}_t$ is the current point.
        - $\mathbf{x}'$ is the new point.
        - $T$ is the temperature.
        - $\Delta E = f(\mathbf{x}') - f(\mathbf{x}_t)$ is the change in function value.
        - $\exp\left(\frac{-\Delta E}{T}\right)$ is the acceptance probability.
        """)

    non_convex_path_sa, lowest_point = simulated_annealing(non_convex_function, non_convex_start, temp, cooling_rate, n_iter)

    # Visualizing the path with alpha changing based on iteration
    alphas = np.linspace(0.1, 1, len(non_convex_path_sa))
    fig_sa = plot_3d_surface(non_convex_function, non_convex_path_sa, "Non-Convex Function (SA)", alphas=alphas, lowest_point=lowest_point)
    
    # Adding blue points for other iteration's observed minimums
    other_mins = non_convex_path_sa[:-1]
    fig_sa.add_trace(go.Scatter3d(
        x=other_mins[:, 0],
        y=other_mins[:, 1],
        z=non_convex_function(other_mins[:, 0], other_mins[:, 1]),
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Observed Minima'
    ))
    
    # Adding the final minimum point in red
    fig_sa.add_trace(go.Scatter3d(
        x=[lowest_point[0]],
        y=[lowest_point[1]],
        z=[non_convex_function(lowest_point[0], lowest_point[1])],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Lowest Observed'
    ))
    
    # Adding the starting point in green
    fig_sa.add_trace(go.Scatter3d(
        x=[non_convex_path_sa[0, 0]],
        y=[non_convex_path_sa[0, 1]],
        z=[non_convex_function(non_convex_path_sa[0, 0], non_convex_path_sa[0, 1])],
        mode='markers',
        marker=dict(size=6, color='green'),
        name='Start'
    ))

    st.plotly_chart(fig_sa)
