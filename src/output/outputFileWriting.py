import os

from setup.configProcessor import get_result_dir


def write_predictions_to_file(predictions):
    averaged_probabilities = {}
    for protein_id in predictions.keys():
        prediction_list = predictions[protein_id]
        averaged_probability = sum(prediction_list)/len(prediction_list)
        averaged_probabilities[protein_id] = averaged_probability

    result_dir = get_result_dir()
    prediction_folder = os.path.join(result_dir, "predictions")
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)

    for protein_id, prediction in averaged_probabilities.items():
        file_name = os.path.join(prediction_folder, f"{protein_id}.bindNode_out")
        with open(file_name, 'w') as fh:
            fh.write("Position\tMetal.RI\tMetal.Class\tNuc.RI\tNuc.Class\tSmall.RI\tSmall.Class\tAny.Class\n")
            for position, residue_prediction in enumerate(prediction):
                metal_prediction = residue_prediction[0] >= 0.5
                small_prediction = residue_prediction[1] >= 0.5
                nuclear_prediction = residue_prediction[2] >= 0.5
                any_prediction = any([metal_prediction, nuclear_prediction, small_prediction])
                fh.write(f"{position}\t")
                fh.write(f"-\t") # TODO: print RI metal instead
                fh.write("b\t" if metal_prediction else "nb\t")
                fh.write(f"-\t")  # TODO: print RI nuclear instead
                fh.write("b\t" if nuclear_prediction else "nb\t")
                fh.write(f"-\t")  # TODO: print RI small instead
                fh.write("b\t" if small_prediction else "nb\t")
                fh.write("b\n" if any_prediction else "nb\n")
