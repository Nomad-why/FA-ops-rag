from tqdm import tqdm
import time
from PIL import Image
from io import BytesIO
from edu_ocr import *
import numpy as np
ocr = get_ocr()

# # 创建一个进度条
# pbar = tqdm(total=100)
#
# for i in range(10):
#     time.sleep(0.5)  # 模拟某个耗时操作
#     pbar.update(10)  # 更新进度
#     if i == 5:
#         pbar.set_description("更新进度")  # 更新描述
#         # pbar.refresh()  # 刷新显示
#
# pbar.close()
from pptx import Presentation


def ppt2text(filepath):
    # 打开指定路径的 PowerPoint 文件
    prs = Presentation(filepath)
    resp = ''
    for slide_number, slide in enumerate(prs.slides, start=1):
        print(f'slide-->{slide}')
        sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))
        for a in slide.shapes:
            print(a.top)
            print(a.left)
        print(f'sorted_shapes--》{sorted_shapes}')
        for shape in sorted_shapes:
            print(f'shape--》{shape}')
            print(f'shape--》{shape.has_text_frame}')
            if shape.has_text_frame:
                # 将文本框中的文本添加到resp中，并去掉前后空格
                resp += shape.text.strip() + "\n"
                print(f'resp000--》{resp}')
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            resp += paragraph.text.strip() + "\n"
                print(f'resp111--》{resp}')
            if shape.shape_type == 13:  # 13 表示图片
                image = Image.open(BytesIO(shape.image.blob))
                result, _ = ocr(np.array(image))
                if result:
                    ocr_result = [line[1] for line in result]
                    resp += "\n".join(ocr_result)
                print(f'resp222--》{resp}')
            elif shape.shape_type == 6:  # 6 表示组合
                print(f'da')
        print('*'*80)
if __name__ == '__main__':
    ppt2text(filepath='测试ppt.pptx')