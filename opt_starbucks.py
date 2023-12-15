from pyomo.environ import *
import random
from pyomo.environ import *
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np
import re
def extract_numeric_order(label):
    match = re.search(r'\d+', label)
    if match:
        return int(match.group())
    return 0  # Return 0 if no numeric part found

def report():
    # Extract solution values
    order_labels = [f'Order {i}' for i in model.v]
    arrival_time_values = [model.arrival[i] for i in model.v]
    start_time_values = [model.start[i].value for i in model.v]
    wait_time_values = [model.wait[i].value for i in model.v]
    finish_time_values = [model.start[i].value + model.process[i] for i in model.v]

    # Sort orders by start time
    sorted_order_data = sorted(zip(order_labels,
                                   arrival_time_values,
                                   start_time_values,
                                   wait_time_values,
                                   finish_time_values), key=lambda x: x[2])

    sorted_idle_time = []
    for i, data in enumerate(sorted_order_data):
        if i == 0:
            sorted_idle_time.append(sorted_order_data[0][2])
        else:
            sorted_idle_time.append(sorted_order_data[i][2]-sorted_order_data[i-1][4])

    sorted_order_labels, sorted_arrival_time, sorted_start_time, sorted_wait_time, sorted_finish_time = zip(*sorted_order_data)

    sorted_order_data1 = sorted(zip(sorted_order_labels,
                                   sorted_arrival_time,
                                   sorted_start_time,
                                   sorted_wait_time,
                                   sorted_finish_time,
                                   sorted_idle_time),  key=lambda x: (x[1], extract_numeric_order(x[0])))

    sorted_order_labels, \
        sorted_arrival_time, \
        sorted_start_time, sorted_wait_time, sorted_finish_time, sorted_idle_time = zip(*sorted_order_data1)

    # Create a Gantt chart
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(len(sorted_order_data1)):
        order_label = sorted_order_labels[i]
        arrival_time = sorted_arrival_time[i]
        start_time = sorted_start_time[i]
        wait_time = sorted_wait_time[i]
        finish_time = sorted_finish_time[i]
        idle_time = sorted_idle_time[i]

        # Plot idle time box
        ax.barh(y=order_label, left=start_time - idle_time, width=idle_time, label='Idle', color='gray')
        # Plot wait time box
        ax.barh(y=order_label, left=arrival_time, width=wait_time, label='Wait', color='yellow')
        # Plot processing time box
        ax.barh(y=order_label, left=arrival_time + wait_time, width=finish_time - start_time, label='Process', color='green')

    ax.invert_yaxis()  # Reverse the y-axis
    #ax.xaxis.set_label_position('top')

    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Orders', fontsize=14)
    ax.set_title('Gantt Chart for idle time, wait time, and service time')
    ax.legend(['Idle', 'Wait', 'Process'], loc='upper right')  # Add legend with labels and place it in the upper right corner
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.savefig(f'plots/gant_chart_example1')
    plt.show()

    # Print the results and generate the report
    print("Objective Value:", value(model.obj))
    print("\nX Variables:")
    for (i, j) in model.e:
        if model.x[i, j].value > 0:
            print(f"x[{i},{j}] = {model.x[i, j].value}")

    print("\nPsi Variables:")
    for i in model.v:
        print(f"psi[{i}] = {model.psi[i].value}")

    print("\nStart Time Variables:")
    for i in model.v:
        print(f"start[{i}] = {model.start[i].value}")

    print("\nWait Time Variables:")
    for i in model.v:
        print(f"wait[{i}] = {model.wait[i].value}")

    # Generate the report (you can save it to a file or print it)
    with open(f'report_{time.time()}.txt', 'w') as report_file:
        report_file.write("Objective Value: " + str(value(model.obj)) + "\n")
        report_file.write("\nX Variables:\n")
        for (i, j) in model.e:
            if model.x[i, j].value > 0:
                report_file.write(f"x[{i},{j}] = {model.x[i, j].value}\n")

        report_file.write("\nPsi Variables:\n")
        for i in model.v:
            report_file.write(f"psi[{i}] = {model.psi[i].value}\n")

        report_file.write("\nStart Time Variables:\n")
        for i in model.v:
            report_file.write(f"start[{i}] = {model.start[i].value}\n")

        report_file.write("\nWait Time Variables:\n")
        for i in model.v:
            report_file.write(f"wait[{i}] = {model.wait[i].value}\n")

    print(f"Report generated as 'report_{time.time()}.txt'")

# Example 1 First-in, First-out (FIFO) is not always optimal, even though it’s a good baseline strategy

#process = {1:300, 2:30, 3:20, 4:10}
#arrival = {1:0, 2:0, 3:60, 4:60}

process = {1: 400, 2: 10, 3: 10, 4: 100, 5: 200, 6: 50}
arrival = {1: 0, 2: 60, 3: 120, 4: 120, 5: 180, 6: 240}

# # Example 2
# process = {1: 25,
#            2: 20,
#            3: 30,
#            4: 20,
#            5: 12,
#            6: 10,
#            7: 22,
#            8: 25,
#            9: 20,
#            10: 10,
#            11: 12,
#            12: 35,
#            13: 12,
#            14: 20,
#            15: 20}
# arrival = {1: 0,
#            2: 0,
#            3: 0,
#            4: 0,
#            5: 0,
#            6: 30,
#            7: 330,
#            8: 330,
#            9: 330,
#            10: 330,
#            11: 390,
#            12: 390,
#            13: 390,
#            14: 390,
#            15: 510}

start_time = time.time()
# Define the set of nodes and edges
num_orders = len(process)
order_set = range(1, num_orders+1)
origin_node = 0
destination_node=len(order_set)+1
edge_set = [(i, j) for i in order_set for j in order_set if i != j]
edge_set.extend([(origin_node, j) for j in order_set])
edge_set.extend([(i, destination_node) for i in order_set])

# Create random opening costs for each facility
# process = {i: random.uniform(10, 20) for i in order_set}
#arrival = {i: random.uniform(20, 30) for i in order_set}

model = ConcreteModel()

# Sets
model.v = Set(initialize=order_set)
model.e = Set(initialize=edge_set)

#Parameters
model.process = Param(model.v, initialize=process)
model.arrival = Param(model.v, initialize=arrival)

# Define the decision variables
model.wait = Var(model.v, within=NonNegativeReals)
model.start = Var(model.v, within=NonNegativeReals)
model.psi = Var(model.v, within=NonNegativeReals)
model.x = Var(model.e, within=Binary)
model.gamma = Var(model.e, within=NonNegativeReals)

big_M = sum(list(arrival.values())) + sum(list(process.values())) + 1

# Define the objective function
model.obj = Objective(expr=sum(model.wait[i] for i in model.v), sense=minimize)

#Define Constraints
model.Constraint1 = ConstraintList()
for i in model.v:
    model.Constraint1.add(model.wait[i] == model.start[i] - model.arrival[i])
#
model.Constraint2 = ConstraintList()
for i in model.v:
    model.Constraint2.add(
        model.start[i] >= model.arrival[i])

model.Constraint2_prime = ConstraintList()
for i in model.v:
    for k in model.v:
        if k != i:
            model.Constraint2_prime.add(
                model.start[i] >= model.gamma[k, i]+model.x[k, i]*model.process[k])

model.Constraint3 = ConstraintList()
for i in model.v:
    model.Constraint3.add(model.x[i, destination_node] + sum(model.x[i, j] for j in model.v if j != i) == 1)

model.Constraint4 = ConstraintList()
for j in model.v:
    model.Constraint4.add(model.x[origin_node, j] + sum(model.x[i, j] for i in model.v if i != j) == 1)

model.Constraint5 = Constraint(expr=sum(model.x[origin_node, j] for j in model.v) == 1)

model.Constraint6 = Constraint(expr=sum(model.x[i, destination_node] for i in model.v) == 1)

model.Constraint7 = ConstraintList()
for i in model.v:
    for j in model.v:
        if i != j and i >= 2 and j >= 2:
            model.Constraint7.add(
                (1-model.x[i, j])*(len(order_set)-1) >= model.psi[i] - model.psi[j] + 1
                )

model.Constraint8 = ConstraintList()
for i in model.v:
    model.Constraint8.add(model.psi[i] <= len(order_set))

model.Constraint9 = ConstraintList()
for i in model.v:
    model.Constraint9.add(model.psi[i] >= 2)

model.Constraint10 = ConstraintList()
for i in model.v:
    for k in model.v:
        if i != k:
            model.Constraint10.add(model.gamma[k, i] <= big_M*model.x[k, i])

model.Constraint11 = ConstraintList()
for i in model.v:
    for k in model.v:
        if i != k:
            model.Constraint11.add(model.gamma[k, i] <= model.start[k])

model.Constraint12 = ConstraintList()
for i in model.v:
    for k in model.v:
        if i != k:
            model.Constraint12.add(model.gamma[k, i] >= model.start[k]-big_M*(1-model.x[k, i]))

# Example 1
# model.Constraint13 = ConstraintList()
# model.Constraint13.add(model.x[0, 1] == 1)
# model.Constraint13.add(model.x[1, 2] == 1)
# model.Constraint13.add(model.x[2, 3] == 1)
# model.Constraint13.add(model.x[3, 4] == 1)

# model.Constraint13 = ConstraintList()
# for i in model.v:
#     model.Constraint13.add(model.x[i, i+1] == 1)
# model.Constraint13.add(model.x[0, 1] == 1)
# model.Constraint13.add(model.x[15, 16] == 1)

# model.Constraint13 = ConstraintList()
# model.Constraint13.add(model.x[1, 3] == 1)
# model.Constraint13.add(model.x[2, 1] == 1)
# model.Constraint13.add(model.x[3, 10] == 1)
# model.Constraint13.add(model.x[4, 6] == 1)
# model.Constraint13.add(model.x[5, 4] == 1)
# model.Constraint13.add(model.x[6, 2] == 1)
# model.Constraint13.add(model.x[7, 8] == 1)
# model.Constraint13.add(model.x[8, 11] == 1)
# model.Constraint13.add(model.x[9, 7] == 1)
# model.Constraint13.add(model.x[10, 9] == 1)
# model.Constraint13.add(model.x[11, 13] == 1)
# model.Constraint13.add(model.x[12, 15] == 1)
# model.Constraint13.add(model.x[13, 14] == 1)
# model.Constraint13.add(model.x[14, 12] == 1)
# model.Constraint13.add(model.x[0, 5] == 1)
# model.Constraint13.add(model.x[15, 16] == 1)

# model.Constraint13 = ConstraintList()
# model.Constraint13.add(model.x[0, 1] == 1)
# model.Constraint13.add(model.x[1, 2] == 1)
# model.Constraint13.add(model.x[2, 3] == 1)
# model.Constraint13.add(model.x[3, 4] == 1)
# model.Constraint13.add(model.x[4, 5] == 1)
# model.Constraint13.add(model.x[5, 6] == 1)
# model.Constraint13.add(model.x[6, 7] == 1)
# model.Constraint13.add(model.x[7, 8] == 1)
# model.Constraint13.add(model.x[8, 9] == 1)
# model.Constraint13.add(model.x[9, 10] == 1)
# model.Constraint13.add(model.x[10, 11] == 1)
# model.Constraint13.add(model.x[11, 12] == 1)
# model.Constraint13.add(model.x[12, 13] == 1)
# model.Constraint13.add(model.x[13, 14] == 1)
# model.Constraint13.add(model.x[14, 15] == 1)
# model.Constraint13.add(model.x[15, 16] == 1)

# My example
# model.Constraint13 = ConstraintList()
# model.Constraint13.add(model.x[0, 1] == 1)
# model.Constraint13.add(model.x[1, 2] == 1)
# model.Constraint13.add(model.x[2, 3] == 1)
# model.Constraint13.add(model.x[3, 4] == 1)
# model.Constraint13.add(model.x[4, 5] == 1)
# model.Constraint13.add(model.x[5, 6] == 1)

model.Constraint13 = ConstraintList()
model.Constraint13.add(model.x[0, 2] == 1)
model.Constraint13.add(model.x[2, 3] == 1)
model.Constraint13.add(model.x[3, 6] == 1)
model.Constraint13.add(model.x[6, 4] == 1)
model.Constraint13.add(model.x[4, 5] == 1)
model.Constraint13.add(model.x[5, 1] == 1)

lp_file = "my_model.lp"  # You can change the filename as needed

# Save the model as an LP file
model.write(lp_file)
# Solve the model
solver = SolverFactory('cbc')
solver.options['ratioGap'] = 0.1
#solver.options['sec'] = 500
solver.options['presolve'] = True
solver.options['threads'] = 4
solver.options['log'] = True
solver.options['heuristics'] = True
#solver.options['barrier'] = True
# results = solver.solve(model)
results = solver.solve(model)
end_time = time.time()

print(f"time is {end_time - start_time}")

if results.solver.termination_condition == TerminationCondition.optimal:
    report()


