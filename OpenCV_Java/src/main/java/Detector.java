import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect;

import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.cvRectangle;


/**
 * Пример кода распознавания обьектов.
 */
public class Detector {

    private static final String CASCADE_FILENAME = "C:\\learn\\cascade.xml";
    private static final int CV_LOAD_IMAGE_GRAYSCALE = 16;
    private static  opencv_objdetect.CascadeClassifier classifier = new opencv_objdetect.CascadeClassifier(CASCADE_FILENAME);

    private static boolean handleImage(String srcFName, String resultFName) {
        //читаем изображение из файла
        opencv_core.Mat mat = imread(srcFName, CV_LOAD_IMAGE_GRAYSCALE);
        //сюда складываем найденное
        opencv_core.RectVector rectVector = new opencv_core.RectVector();
        classifier.detectMultiScale(mat, rectVector); //поиск фрагментов
        boolean hasFound = rectVector.size() > 0;
        if (hasFound) {
            //открываем снова изображение для рисования
            opencv_core.IplImage src = cvLoadImage(srcFName, 0);
            for (int i = 0; i <= rectVector.size(); i++) {
                opencv_core.Rect rect = rectVector.get(i);
                int height = rect.height();
                int width = rect.width();
                int x = rect.tl().x();
                int y = rect.tl().y();
                opencv_core.CvPoint start = new opencv_core.CvPoint(x, y);
                opencv_core.CvPoint finish = new opencv_core.CvPoint(x + width, y + height);
                //обводим объект в прямоугольник
                cvRectangle(src, start, finish, opencv_core.CvScalar.RED, 2, 8, 0);
            }
            cvSaveImage(resultFName, src); //сохраняем копию с обведенными объектами
        }
        return hasFound;
    }
}
