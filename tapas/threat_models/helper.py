import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

def plot_roc(y, y_score):
    # print(y)
    # print(y_score)
    fpr, tpr, thresholds = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    # Create the ROC curve trace
    roc_trace = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})',
        line=dict(color='blue', width=2)
    )

    # Create the diagonal line trace
    diagonal_trace = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', dash='dash')
    )

    # Create the layout
    layout = go.Layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        showlegend=True
    )

    # Create the figure and add traces
    fig = go.Figure(data=[roc_trace, diagonal_trace], layout=layout)

    # Return the Plotly figure
    return fig
