"""
created on:2022/10/30 23:13
@author:caijianfeng
"""
import json
import os
import cv2


def get_predict_result(img_dir, predict_dir):
    json_file = os.path.join(img_dir, 'config.json')
    with open(json_file, 'r') as f:
        imgs_anns = json.load(f)
    predict_file = os.path.join(predict_dir, 'coco_instances_results.json')
    with open(predict_file, 'r') as f:
        predict_anns = json.load(f)
    
    for predict_ann in predict_anns:
        img_id = predict_ann['image_id']
        img_category = predict_ann['category_id']
        img_bbox = predict_ann['bbox']
        # print(type(imgs_anns.values()))
        img_ann = list(imgs_anns.values())[img_id]
        filename = img_ann['img_path'][1:].replace('\\', '/')
        bbox = [img_ann['boy_y'][0], img_ann['box_x'][0], img_ann['boy_y'][1], img_ann['box_x'][1]]
        category = img_ann['category']
        bboxs = [bbox, img_bbox]
        # 如果预测类别正确, 再展示预测框
        if img_category == category:
            save_filename = os.path.join(predict_dir + '/category_correct', str(img_id) + '.png')
            print(save_filename)
            names = ['true bbox', 'predict bbox']
            print('predict category:', img_category, '; true category:', category)
            print('predict bbox:', img_bbox)
            print('true bbox:', bbox)
            draw_rectangle_by_point(img_file_path=filename,
                                    names=names,
                                    new_img_file_path=save_filename,
                                    bboxs=bboxs)
        else:
            save_filename = os.path.join(predict_dir + '/category_uncorrect', str(img_id) + '.png')
            names = [str(img_category), str(category)]
            draw_rectangle_by_point(img_file_path=filename,
                                    names=names,
                                    new_img_file_path=save_filename,
                                    bboxs=bboxs)


# Plot the predicted box against the true box
def draw_rectangle_by_point(img_file_path, names, new_img_file_path, bboxs):
    image = cv2.imread(img_file_path)
    for i, bbox in enumerate(bboxs):
        first_point = (int(bbox[0]), int(bbox[1]))
        last_point = (int(bbox[2]), int(bbox[3]))
        
        print("左上角：", first_point)
        print("右下角：", last_point)
        cv2.rectangle(image, first_point, last_point, (0, 255, 0), 1)  # Draw a box on the image
        cv2.putText(image, names[i], first_point, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0),
                    thickness=1)  # Draw the name of the box above the rectangle
    cv2.imshow('predict', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite(new_img_file_path, image)


if __name__ == '__main__':
    img_dir = './dataset'
    predict_dir = './output'
    get_predict_result(img_dir, predict_dir)
