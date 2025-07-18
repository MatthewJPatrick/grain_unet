import matplotlib.pyplot as plt
import pandas as pd
def plot_loss_points(loss_points_training, loss_point_validation, interimloss_point, filename_training='lossPlotTraining.png',filename_validation='lossPlotValidation'):
    '''
Function that takes in an array of loss function values and exports a png at the end of the session
'''

    # Create a new figure
    plt.figure()
    
    # Plot data points
    plt.plot(loss_points_training, marker='o', linestyle='-')
    
    # Add title and labels
    plt.title('Loss over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Fn Value (Training)')
    
    # Save the plot as a PNG file
    plt.savefig(filename_training)
    plt.close()  # Close the figure

    #validation figure creation
    plt.figure()

     # Plot data points
    plt.plot(loss_point_validation, marker='o', linestyle='-')
    
    # Add title and labels
    plt.title('Loss over Validation Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Fn Value (Validation)')
    
    # Save the plot as a PNG file
    plt.savefig(filename_validation)
    plt.close()  # Close the figure

    #[bookmark potential]
    #storing data for external plotting
    filename_training_csv = "training_loss_graph_data.csv"
    filename_validation_csv = "validtation_loss_graph_data.csv"
    filename_interimlosses_csv = "interimlosses_for_Rickman.csv"

    df_training_loss_point = pd.DataFrame(loss_points_training, columns=['Epochs'])

    df_validation_loss_point = pd.DataFrame(loss_point_validation, columns=['Epochs'])

    df_interimloss_for_Rickman = pd.DataFrame(interimloss_point, columns=['Epochs', 'Interim Loss'])

    df_training_loss_point.to_csv(filename_training_csv, index=False)
    
    df_validation_loss_point.to_csv(filename_validation_csv,index=False)

    df_interimloss_for_Rickman.to_csv(filename_interimlosses_csv, index=False)

