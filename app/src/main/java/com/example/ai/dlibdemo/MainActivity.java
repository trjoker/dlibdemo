package com.example.ai.dlibdemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.media.ExifInterface;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    // Storage Permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    private Button detectBtn;
    private Button makeUpBtn;
    private Button restoreBtn;
    private ImageView imageView;
    private Bitmap bitmap;
    //这里添加模板图片
    private Bitmap eye_left;
    private Bitmap eye_right;

    // add eyebrow Bitmap obj
    private Bitmap eye_brow_left;
    private Bitmap eye_brow_right;

    private Bitmap blush_l;
    private Bitmap blush_r;

    private Button lipBtn;
    private Button eyeshadowBtn;
    private Button eyeBrowBtn;
    private Button beautyBtn;
    private Button fastBtn;
    private Button lashBtn;
    private Button blusherBtn;
    private Button eyeLinearBtn;
    private Button liftBtn;

    //加速检测
    private boolean isFast;
    FaceDet faceDet;
    List<VisionDetRet> results;
    List<Point> points;

    private static int MAKEUP = 1;
    private static int LIP = 2;
    private static int EYESHADOW = 3;
    private static int EYEBROW = 4;
    private static int BEAUTY = 5;
    private static int LASH = 6;
    private static int BLUSHER = 7;
    private static int EYELINEAR = 8;
    private static int LIFT = 9;


    //图片大小
    private int height = 1800;
    private int width = 1200;


    //加速检测时 图片的缩放倍数
    private static int SCALE = 5;

    private int scale = 1;

    //人脸框
    int rectLeft;
    int rectTop;
    int rectRight;
    int rectBottom;
    public static final BitmapFactory.Options OPTION_RGBA8888 = new BitmapFactory.Options();
    public static final BitmapFactory.Options OPTION_A8 = new BitmapFactory.Options();

    static {
        // Android's Bitmap.Config.ARGB_8888 is misleading, its memory layout is RGBA, as shown in
        // JNI's macro ANDROID_BITMAP_FORMAT_RGBA_8888, and getPixel() returns ARGB format.
        OPTION_RGBA8888.inPreferredConfig = Bitmap.Config.ARGB_8888;
        OPTION_RGBA8888.inDither = false;
        OPTION_RGBA8888.inMutable = true;
        OPTION_RGBA8888.inPremultiplied = false;

        OPTION_A8.inPreferredConfig = Bitmap.Config.ALPHA_8;
        OPTION_A8.inDither = false;
        OPTION_A8.inMutable = true;
        OPTION_A8.inPremultiplied = false;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        verifyStoragePermissions(this);
        // Example of a call to a native method
        initView();
        initEvent();
        initValue();
        initFaceDet();
    }

    private void initEvent() {
        detectBtn.setOnClickListener(this);
        makeUpBtn.setOnClickListener(this);
        restoreBtn.setOnClickListener(this);
        lipBtn.setOnClickListener(this);
        eyeshadowBtn.setOnClickListener(this);
        eyeBrowBtn.setOnClickListener(this);
        beautyBtn.setOnClickListener(this);
        fastBtn.setOnClickListener(this);
        lashBtn.setOnClickListener(this);
        blusherBtn.setOnClickListener(this);
        eyeLinearBtn.setOnClickListener(this);
        liftBtn.setOnClickListener(this);
    }

    private void initView() {
        lipBtn = findViewById(R.id.btn_lip);
        eyeshadowBtn = findViewById(R.id.btn_eye_shadow);
        eyeBrowBtn = findViewById(R.id.btn_eye_brow);
        beautyBtn = findViewById(R.id.btn_beauty);
        fastBtn = findViewById(R.id.btn_fast);
        detectBtn = findViewById(R.id.btn_detection);
        makeUpBtn = findViewById(R.id.btn_makeup);
        restoreBtn = findViewById(R.id.btn_restore);
        lashBtn = findViewById(R.id.btn_lash);
        blusherBtn = findViewById(R.id.btn_blusher);
        eyeLinearBtn = findViewById(R.id.btn_eye_linear);
        liftBtn = findViewById(R.id.btn_lift);
        imageView = findViewById(R.id.image);

        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.model);
        //这里载入模板图片，并锁定图片为原图大小
        BitmapFactory.Options bfo = new BitmapFactory.Options();
        bfo.inScaled = false;
        eye_left = BitmapFactory.decodeResource(getResources(), R.drawable.zuo3, bfo);
        eye_right = BitmapFactory.decodeResource(getResources(), R.drawable.you3, bfo);

        eye_brow_left = BitmapFactory.decodeResource(getResources(), R.drawable.eyebrow_zuo, bfo);
        eye_brow_right = BitmapFactory.decodeResource(getResources(), R.drawable.eyebrow_you, bfo);

        blush_l = BitmapFactory.decodeResource(getResources(), R.drawable.blush_l, bfo);
        blush_r = BitmapFactory.decodeResource(getResources(), R.drawable.blush_r, bfo);

        //压缩图片否则会oom
        bitmap = Bitmap.createScaledBitmap(bitmap, width, height, true);
        imageView.setImageBitmap(bitmap);
    }

    private void initValue() {
        points = new ArrayList<>();

    }

    public static void verifyStoragePermissions(Activity activity) {

        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,
                        REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void showImage() {
        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.model);
        imageView.setImageBitmap(bitmap);
    }


    //将android point 转化为 opencv point
    private Point changePoint(android.graphics.Point point) {
        return new Point(point.x, point.y);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_detection:
                detection();
                break;
            case R.id.btn_makeup:
                makeUp(MAKEUP);
                break;
            case R.id.btn_lip:
                makeUp(LIP);
                break;
            case R.id.btn_eye_shadow:
                makeUp(EYESHADOW);
                break;
            case R.id.btn_eye_brow:
                makeUp(EYEBROW);
                break;
            case R.id.btn_beauty:
                makeUp(BEAUTY);
                break;
            case R.id.btn_lash:
                makeUp(LASH);
                break;
            case R.id.btn_blusher:
                makeUp(BLUSHER);
                break;
            case R.id.btn_eye_linear:
                makeUp(EYELINEAR);
                break;
            case R.id.btn_lift:
                makeUp(LIFT);
                break;

            case R.id.btn_restore:
                restore();
                break;
            case R.id.btn_fast:
                if (!isFast) {
                    fastBtn.setText("加速检测开");
                    isFast = true;
                    scale = SCALE;
                } else {
                    fastBtn.setText("加速检测关");
                    isFast = false;
                    scale = 1;
                }
                break;
        }
    }


    //还原显示
    private void restore() {
        imageView.setImageBitmap(bitmap);
    }

    //调用化妆功能
    private void makeUp(int type) {
        if (points.isEmpty()) {
            Toast.makeText(this, "还未检测特征点", Toast.LENGTH_SHORT).show();
            return;
        }
        Long start = System.currentTimeMillis();
        //修改w,h，所有模板长宽存入数组
        // add eyebrow left,right w,h;
        int w[] = {bitmap.getWidth(), eye_left.getWidth(), eye_right.getWidth(), eye_brow_left
                .getWidth(), eye_brow_right.getWidth(),blush_l.getWidth(),blush_r.getWidth()};
        int h[] = {bitmap.getHeight(), eye_left.getHeight(), eye_right.getHeight(), eye_brow_left
                .getHeight(), eye_brow_right.getHeight(),blush_l.getHeight(),blush_r.getHeight()};
        int[] piexls = new int[w[0] * h[0]];
        //这里添加模板piexl
        // add eyebrow piexl;
        int[] p_eye_left = new int[w[1] * h[1]];
        int[] p_eye_right = new int[w[2] * h[2]];

        int[] p_eyebrow_left = new int[w[3] * h[3]];
        int[] p_eyebrow_right = new int[w[4] * h[4]];

        int[] p_blush_l = new int[w[5] * h[5]];
        int[] p_blush_r = new int[w[6] * h[6]];

        bitmap.getPixels(piexls, 0, w[0], 0, 0, w[0], h[0]);
        eye_left.getPixels(p_eye_left, 0, w[1], 0, 0, w[1], h[1]);
        eye_right.getPixels(p_eye_right, 0, w[2], 0, 0, w[2], h[2]);

        // get eyebrow pixel
        eye_brow_left.getPixels(p_eyebrow_left, 0, w[3], 0, 0, w[3], h[3]);
        eye_brow_right.getPixels(p_eyebrow_right, 0, w[4], 0, 0, w[4], h[4]);

        blush_l.getPixels(p_blush_l, 0, w[5], 0, 0, w[5], h[5]);
        blush_r.getPixels(p_blush_r, 0, w[6], 0, 0, w[6], h[6]);

        int[] pointsArray = new int[points.size() * 2];

        for (int i = 0; i < points.size(); i++) {
            pointsArray[2 * i] = points.get(i).x * scale;
            pointsArray[2 * i + 1] = points.get(i).y * scale;
        }

//        Bitmap lashDst = decodeFile("/storage/emulated/0/Pictures/test.png",
//                OPTION_RGBA8888);
//
//        Bitmap lashSrc = lashDst.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap lashMask = BitmapFactory.decodeResource(getResources(), R.drawable
                .eye_lash_00, OPTION_A8);

        //修改调用
        int[] resultData = Bitmap2proc(piexls, p_eye_left, p_eye_right, p_eyebrow_left,
                p_eyebrow_right,p_blush_l,p_blush_r, lashMask, w, h, pointsArray, type,
                rectLeft, rectTop,
                rectRight, rectBottom);
        Bitmap resultImage = Bitmap.createBitmap(w[0], h[0], Bitmap.Config.ARGB_8888);
        resultImage.setPixels(resultData, 0, w[0], 0, 0, w[0], h[0]);
        imageView.setImageBitmap(resultImage);
        Long end = System.currentTimeMillis();
        Toast.makeText(MainActivity.this, "化妆完成  耗时" + (end - start) + "ms", Toast.LENGTH_LONG)
                .show();
        Log.i("taoran", "化妆完成 耗时" + (end - start) + "ms");
    }


    //检测特征点
    private void detection() {
        if (faceDet == null) {
            Toast.makeText(this, "模型还未初始化好", Toast.LENGTH_SHORT).show();
            return;
        }
        Bitmap bitmap2 = Bitmap.createScaledBitmap(bitmap, width / scale, height / scale, true);

        Long start = System.currentTimeMillis();
        if (bitmap != null) {
            results = faceDet.detect(bitmap2);
            for (final VisionDetRet ret : results) {
                points = ret.getFaceLandmarks();
                rectLeft = ret.getLeft() * scale;
                rectTop = ret.getTop() * scale;
                rectRight = ret.getRight() * scale;
                rectBottom = ret.getBottom() * scale;
            }
        }
        Long end = System.currentTimeMillis();
        if (points.isEmpty()) {
            Toast.makeText(this, "检测失败", Toast.LENGTH_SHORT).show();
            return;
        }
        Toast.makeText(MainActivity.this, "检测特征点完成 耗时" + (end - start) + "ms", Toast.LENGTH_LONG)
                .show();
        Log.i("taoran", "检测特征点完成  耗时" + (end - start) + "ms");

    }

    @Override
    public void onResume() {
        super.onResume();
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native int[] Bitmap2proc(int[] pixels, int[] eye_left, int[] eye_right,
                                    int[] p_eyebrow_left, int[] p_eyebrow_right, int[] blush_l, int[] blush_r, Bitmap lash,
                                    int[] w, int[] h, int[] points, int type, int left, int top,
                                    int right, int bootom);


    //初始化人脸检测模型
    private void initFaceDet() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                faceDet = new FaceDet(Constants.getFaceShapeModelPath());
            }
        }).start();
    }

    public static Bitmap decodeFile(@NonNull String pathName, @Nullable BitmapFactory.Options
            opts) {
        Bitmap bitmap = BitmapFactory.decodeFile(pathName, opts);

        final String lowerPathName = pathName.toLowerCase();
        boolean isJPEG = lowerPathName.endsWith(".jpg") || lowerPathName.endsWith(".jpeg");
        if (isJPEG) {
            Matrix matrix = getOrientationFromJPEGFile(pathName);
            Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap
                    .getHeight(), matrix, true);

            bitmap.recycle();
            return rotated;
        }

        return bitmap;
    }

    public static Matrix getOrientationFromJPEGFile(String path) {
        int orientation = ExifInterface.ORIENTATION_UNDEFINED;
        Matrix matrix = new Matrix();  // identity matrix
        try {
            ExifInterface exif = new ExifInterface(path);
            orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface
                    .ORIENTATION_UNDEFINED);
        } catch (IOException e) {
            e.printStackTrace();
            return matrix;
        }

        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                break;
            case ExifInterface.ORIENTATION_NORMAL:
//			return bitmap;
//			[[fallthrough]];
            default:
                break;
        }

        return matrix;
    }
}