from controller import Controller


def main(pls_file_path):
    images_lst = []
    main_path = pls_file_path.replace(pls_file_path.split("/")[-1], "")
    with open(pls_file_path) as pls_file:
        images_paths = pls_file.readlines()
        for img_path in images_paths[2:]:
            images_lst.append(main_path + img_path.replace("\n", ""))
        pkl_file_path = main_path + images_paths[0].replace("\n", "")
    controller = Controller(pkl_file_path, images_lst)
    controller.run()


if __name__ == '__main__':
    main("../resource/play_list.pls")
