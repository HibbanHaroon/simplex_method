import streamlit as st
import numpy as np
from scipy.optimize import linprog
import pandas as pd
import matplotlib.pyplot as plt

def maximize_profit():
    st.header("Maximize Profit: Chairs and Tables Production")

    st.markdown(
        "A company manufactures two types of products: chairs and tables.  \n"
        "The company has limited resources of labor and raw materials. "
        "They want to maximize their profit by deciding how many chairs and tables to produce while considering the constraints of available resources."
    )

    st.write(
        "Let's define the following:  \n  \n"

        "- Each chair earns a profit of \$20, and each table earns a profit of $30."
        "\n- To manufacture a chair, it requires 2 hours of labor and 1 unit of raw material."
        "\n- To manufacture a table, it requires 3 hours of labor and 2 units of raw material."
        "\n- The company has 100 hours of labor and 60 units of raw material available."
    )

    st.subheader("Problem Definition:")
    st.write(
        "**Objective function:**  \n" 
        "Maximize: $P = 20x + 30y$  \n"
        "\n where:"
        "\n- x is the number of chairs produced."
        "\n- y is the number of tables produced."
    )
    st.write("**Subject to constraints:**")
    st.markdown(
        "- Labor constraint: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $2x + 3y <= 100$"
        "\n- Raw material constraint: &nbsp;&nbsp;&nbsp;&nbsp; $x + 2y <= 60$"
        "\n- Non-negativity constraint: &nbsp;&nbsp;&nbsp;&nbsp; $x, y >= 0$"
    )

    # Solving the linear programming
    # Objective function coefficients
    c = [-20, -30]  
    A = [
        [2, 3],  # Labor constraint coefficients
        [1, 2]   # Raw material constraint coefficients
    ]
    # Available resources
    b = [100, 60]  
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

    if result.success:
        st.subheader("Solution:")
        st.write(
            f"Optimal number of chairs (x): {result.x[0]:.2f}  \n"
            f"Optimal number of tables (y): {result.x[1]:.2f}"
        )

        st.markdown("**These values also satisfied our constraint :**")
        st.write(f"Labor Constraint : $2x + 3y <= 100$ : $2({result.x[0]:.2f}) + 3({result.x[1]:.2f}) <= 100$")
        st.write(f"Material Constraint : $x + 2y <= 60$ : ${result.x[0]:.2f} + 2({result.x[1]:.2f}) <= 60$")

        # Maximize profit calculation using obtained values
        max_profit = 20 * result.x[0] + 30 * result.x[1]
        st.subheader("Maximizing Profit:")
        st.write(
            f"Maximum profit calculated: $20 * ({result.x[0]:.2f}) + 30 * ({result.x[1]:.2f}) = $ \${max_profit}  \n"
            f"Maximum profit obtained: \${-result.fun}" # negative sign due to maximizing
        )  
    else:
        st.error("No solution found.")

    # Visualize the constraints and highlight optimal solution
    st.subheader("Visualization:")
    plt.figure(figsize=(8, 6))

    # Plotting the labor constraint line
    x_vals = np.linspace(0, 40, 100)
    y_vals_labor = (100 - 2 * x_vals) / 3
    plt.plot(x_vals, y_vals_labor, label='Labor Constraint: 2x + 3y <= 100')

    # Plotting the material constraint line
    y_vals_material = (60 - x_vals) / 2
    plt.plot(x_vals, y_vals_material, label='Material Constraint: x + 2y <= 60')

    # Highlight the optimal solution
    plt.scatter(result.x[0], result.x[1], color='red', label='Optimal Solution')

    plt.xlabel('Number of Chairs (x)')
    plt.ylabel('Number of Tables (y)')
    plt.title('Optimal Solution and Constraints Visualization')
    plt.legend()

    st.pyplot(plt)
    st.write("Both the equations intersects at the optimal solution which is 20 no. of chairs on x-axis and 20 no. of tables on y-axis")

def main():
    st.set_page_config(page_title="Simplex Method", page_icon="📕")

    st.title("Linear Programming and Simplex Method")

    st.subheader("Linear Programming")
    st.write(
        '''Linear Programming is a mathematical method used to maximize or minimize a linear objective function 
        while satisfying a set of linear equality and inequality constraints. 
        It involves finding the best outcome from a set of possible solutions.'''
    )

    st.subheader("Simplex Method")
    st.write(
        '''The Simplex Method is an iterative algorithm used to solve linear programming problems 
        by systematically moving from one feasible solution to another along the edges of the feasible region, 
        seeking the optimum solution. It starts at a feasible vertex and pivots to neighboring vertices, 
        improving the objective function value at each step. The method involves these key steps:'''
    )
    st.write(
        "- **Initialization:** Start from a basic feasible solution.  \n"
        "- **Pivoting:** Iteratively move to adjacent vertices improving the objective function.  \n"
        "- **Optimality Test:** Continue pivoting until an optimal solution is reached.  \n"
    )
    st.write(
        '''The Simplex Method iteratively traverses the edges of the feasible region, improving the objective function value, 
        until it reaches an optimal solution where no further improvement is possible.'''
    )

    maximize_profit()

if __name__ == "__main__":
    main()
