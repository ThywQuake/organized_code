import os


def main():
    print("Hello from organized-code!")
    data_folder = "data/"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        os.makedirs(os.path.join(data_folder, "clean"))
    print(f"Data folder '{data_folder}' is ready.")

    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, "model"))
        os.makedirs(os.path.join(output_folder, "eval"))
        os.makedirs(os.path.join(output_folder, "pred"))
    print(f"Output folder '{output_folder}' is ready.")


if __name__ == "__main__":
    main()
    from giems_lstm.utils.train import _train

    _train(thread_id=0, config_path="config/F.toml", debug=True, para=1, left=True)
