# --------------------------------------------------------
#
# PYTHON PROGRAM DEFINITION
#
# The knowledge a computer has of Python can be specified in 3 levels:
# (1) Prelude knowledge --> The computer has it by default.
# (2) Borrowed knowledge --> The computer gets this knowledge from 3rd party libraries defined by others
#                            (but imported by us in this program).
# (3) Generated knowledge --> The computer gets this knowledge from the new functions defined by us in this program.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer first processes this PYTHON PROGRAM DEFINITION section of the file.
# On it, our computer enhances its Python knowledge from levels (2) and (3) with the imports and new functions
# defined in the program. However, it still does not execute anything.
#
# --------------------------------------------------------

# ------------------------------------------
# IMPORTS
# ------------------------------------------
import sys
import codecs

# ---------------------------------------------
# FUNCTION parse_spark_streaming_solution_file
# ---------------------------------------------
def parse_spark_streaming_solution_file(my_file, initial_intermediate_gap, final_gap):
    # 1. We create the output variable
    res = []

    # 2. We open the file for reading
    my_input_stream = codecs.open(my_file, "r", encoding="utf-8")

    # 3. We create as many auxiliary variables as needed
    num_lines = 0
    num_batches = 0
    batch_line_indexes = []

    # 4. We traverse the lines of the file
    for line in my_input_stream:
        # 4.1. We strip the line and remove any white space on it
        line = line.strip().replace(" ", "")

        # 4.2. If the line is non-empty we consider it
        if (line):
            # 4.2.1. If it is the line representing a batch
            if (line.startswith("Time:")):
                # I. We edit the line
                line = "Batch " + str(num_batches)

                # II. We mark this line as one containing batches
                batch_line_indexes.append(num_lines)

                # III. We increase the batch index for further ones
                num_batches += 1

            # 4.2.2. We append the line
            res.append(line)
            num_lines += 1

    # 4. We close the file
    my_input_stream.close()

    # 5. We find the batches to remove
    batch_to_remove = [ (batch_line_indexes[index] == (batch_line_indexes[index+1] - initial_intermediate_gap)) for index in range(num_batches-1) ]
    batch_to_remove.append( batch_line_indexes[num_batches-1] == num_lines - final_gap )

    # 6. We traverse the batches to remove the desired ones
    for index in range(num_batches-1, -1, -1):
        if (batch_to_remove[index] == True):
            # 6.1. We remove it
            for _ in range(initial_intermediate_gap):
                del res[batch_line_indexes[index] - 1]
            num_lines -= initial_intermediate_gap

    # 7. We rename the batches accordingly
    valid_batch_index = 0
    for index in range(num_lines):
        # 7.1. If the batch is valid
        if ("Batch " in res[index]):
            # 7.1.1. We edit its line
            res[ index ] = "Batch " + str(valid_batch_index)

            # 7.1.2. We increase the number of valid batches
            valid_batch_index += 1

    # 8. We return res
    return res

# ------------------------------------------
# FUNCTION pass_test
# ------------------------------------------
def pass_test(my_file_1, my_file_2, initial_intermediate_gap, final_gap):
    # 1. We create the output variable
    res = True

    # 2. We read the full content of each file, removing any empty lines and spaces
    content_1 = parse_spark_streaming_solution_file(my_file_1, initial_intermediate_gap, final_gap)
    content_2 = parse_spark_streaming_solution_file(my_file_2, initial_intermediate_gap, final_gap)

    # 3. We check that both files are equal
    size_1 = len(content_1)

    # 3.1. If both files have the same length
    if (size_1 == len(content_2)):
        # 3.1.1. We compare them line by line
        for index in range(size_1):
            if (content_1[index] != content_2[index]):
                res = False
                break

    # 3.2. If the files have different lengths then they are definitely not equal
    else:
        res = False

    # 4. We return res
    return res

# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We get the input values
    my_solution = "./my_solution_1.txt"
    expected_solution = "./A02_ex1_spark_streaming.txt"

    # 1.1. If the program is called from console, we modify the parameters
    if (len(sys.argv) > 1):
        # 1.1.1. We get the student folder path
        my_solution = sys.argv[1]

        # 1.1.2. We get the templates path
        expected_solution = sys.argv[2]

    # 2. We hardcode the initial_intermediate_gap and final_gap for Spark Streaming
    # Spark Streaming => An initial/intermediate batch is empty if there are 3 lines to the next batch
    #                    A final batch is empty if its index is FileSize - 2
    # Spark Structured Streaming => An initial/intermediate batch is empty if there are 7 lines to the next batch
    #                    A final batch is empty if its index is FileSize - 5
    initial_intermediate_gap = 3
    final_gap = 2

    # 3. We check if my_solution matches the expected_solution
    res = pass_test(my_solution, expected_solution, initial_intermediate_gap, final_gap)

    # 4. We print whether we pass the test or not
    print(res)
