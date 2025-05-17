from plotly import express as px
import csv
from os import path, makedirs, getcwd
from numpy import array
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def loadMatrixFromCSV(filename):

    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        x = list(reader)
        result = array(x).astype("float")
    return result

def visualizeHeatmap(filename):

    output_dir = path.join(getcwd(), 'images')
    makedirs(output_dir, exist_ok=True)
    image_path = path.join(output_dir, 'Heatmap.png')

    adj_matrix = loadMatrixFromCSV(filename)
    figure = px.imshow(adj_matrix)
    figure.write_image(image_path)
    figure.show()

def visualizeGraph(filename):
    # Load weighted adjacency matrix from CSV
    csv_file = filename
    df = pd.read_csv(csv_file, index_col=0)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with weights
    for src in df.index:
        for dst in df.columns:
            weight = df.loc[src, dst]
            if pd.notna(weight) and weight != 0:
                G.add_edge(src, dst, weight=weight)

    # Draw the graph
    # Alternative Layouts, but shell or circular seem to be the best
        #pos = nx.spring_layout(G, seed=42, k =1.5, iterations=100)  # For consistent layout
        #pos = nx.circular_layout(G)

    pos = nx.shell_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold',
            arrows=True)

    #Edge Labels can be drawn but have been omitted for better readability of the graph and comparison
    #edge_labels = nx.get_edge_attributes(G, 'weight')
    #edge_labels = {
    #    (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)
    #}
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Graph from Weighted Adjacency Matrix")
    plt.axis('off')
    plt.tight_layout()

    output_dir = path.join(getcwd(), 'images')
    makedirs(output_dir, exist_ok=True)
    image_path = path.join(output_dir, 'graph_layout.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    plt.show()