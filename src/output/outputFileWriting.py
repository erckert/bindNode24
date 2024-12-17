import os

from setup.configProcessor import get_result_dir, write_ri, get_cutoff
from misc.enums import LabelType
from data_processing.post_processing import get_averaged_predictions


def write_predictions_to_file(predictions):
    averaged_probabilities = get_averaged_predictions()

    is_write_ri = write_ri()

    result_dir = get_result_dir()
    prediction_folder = os.path.join(result_dir, "predictions")
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)

    for protein_id, prediction in averaged_probabilities.items():
        file_name = os.path.join(prediction_folder, f"{protein_id}.bindNode_out")
        with open(file_name, 'w') as fh:
            if is_write_ri:
                fh.write("Position\tMetal.RI\tMetal.Class\tNuc.RI\tNuc.Class\tSmall.RI\tSmall.Class\tAny.Class\n")
            else:
                fh.write("Position\tMetal.Prob\tMetal.Class\tNuc.Prob\tNuc.Class\tSmall.Prob\tSmall.Class\tAny.Class\n")
            for position, residue_prediction in enumerate(prediction):
                metal_probability = residue_prediction[LabelType.METAL.value]
                small_probability = residue_prediction[LabelType.SMALL.value]
                nuclear_probability = residue_prediction[LabelType.NUCLEAR.value]

                metal_prediction = metal_probability >= get_cutoff()
                small_prediction = small_probability >= get_cutoff()
                nuclear_prediction = nuclear_probability >= get_cutoff()

                any_prediction = any([metal_prediction, nuclear_prediction, small_prediction])

                fh.write(f"{position + 1}\t")
                fh.write(f"{compute_ri(metal_probability):.3f}\t" if is_write_ri else f"{metal_probability:.3f}\t")
                fh.write("b\t" if metal_prediction else "nb\t")
                fh.write(f"{compute_ri(nuclear_probability):.3f}\t" if is_write_ri else f"{nuclear_probability:.3f}\t")
                fh.write("b\t" if nuclear_prediction else "nb\t")
                fh.write(f"{compute_ri(small_probability):.3f}\t" if is_write_ri else f"{small_probability:.3f}\t")
                fh.write("b\t" if small_prediction else "nb\t")
                fh.write("b\n" if any_prediction else "nb\n")


def compute_ri(prediction_probability):
    if prediction_probability < 0.5:
        reliability_index = round((0.5 - prediction_probability) * 9 / 0.5)
    else:
        reliability_index = round((prediction_probability - 0.5) * 9 / 0.5)

    return reliability_index
