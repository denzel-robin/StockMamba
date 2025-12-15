
# StockMamba

StockMamba is a stock price forecasting model developed using Mamba architecture.
A Mamba model is a state space model suitable for doing temporal forecasting.
The Mamba model can perform these tasks in linear time complexity
while it's equivalent transformer model takes quadratic time complexity.
It can performs in such low time complexity because of the technology
it uses called Selective State Space Scan.

You can read more about this model here: <https://arxiv.org/abs/2312.00752>

## Recommended to run in colab or in an isolated python environment

### colab

1. Open the `StockMamba.ipynb` file.

2. Download it.

3. Open `StockMamba.ipynb` file in your google colab.

4. Change the runtime type to T4 GPU in colab.

5. Run all cells.

### Isolated python environment

1. Clone the repository.

     ``` shell
     git clone https://github.com/denzel-robin/StockMamba.git
     cd StockMamba
     ```

2. Create an isolated python environment.

     ``` shell
     python -m venv venv
     ```

3. Activate the environment.

     ``` shell
     source venv/bin/activate.fish
     ```

     **Note: activate file is different for each shell.**

4. Install the dependencies from `requirement.txt`.

5. Run `main.py` file.

     ``` shell
     python main.py
     ```
