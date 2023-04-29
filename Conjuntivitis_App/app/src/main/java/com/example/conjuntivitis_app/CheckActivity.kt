package com.example.conjuntivitis_app

import android.app.AlertDialog
import android.content.Intent
import android.content.Intent.ACTION_GET_CONTENT
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.conjuntivitis_app.ml.LiteModel
import com.example.conjuntivitis_app.ui.theme.Conjuntivitis_AppTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class CheckActivity : ComponentActivity() {
    private lateinit var select_btn: Button
    private lateinit var predict_btn: Button
    private lateinit var camera_btn: Button
    private lateinit var resultView: TextView
    private lateinit var imageView: ImageView
    private lateinit var bitmap: Bitmap
    private lateinit var labels: List<String>
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var okay_btn:Button
    private lateinit var button3:Button
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        val assetManager: AssetManager = applicationContext.assets
        // read labels from file
       val labels = assetManager.open("labels.txt").bufferedReader().readLines()
        //labels= listOf("Red","White")


        setContentView(R.layout.activity_main)
        okay_btn=findViewById(R.id.okay_btn)
        select_btn = findViewById(R.id.select_btn)
        camera_btn = findViewById(R.id.camera_btn)
        predict_btn = findViewById(R.id.predict_btn)
        resultView = findViewById(R.id.resultView)
        imageView = findViewById(R.id.imageView)
        button3=findViewById(R.id.button3)
        var imageProcessor = ImageProcessor.Builder().add(NormalizeOp(0.0f, 255.0f))
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        select_btn.setOnClickListener {
            val intent = Intent()
            intent.action = ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }
        okay_btn.setOnClickListener {
            okay_btn.setOnClickListener {
                val dialogBuilder = AlertDialog.Builder(this)
                val inflater = this.layoutInflater
                val dialogView = inflater.inflate(R.layout.popup_layout, null)
                dialogBuilder.setView(dialogView)

                val alertDialog = dialogBuilder.create()

                val exitImageView = dialogView.findViewById<ImageView>(R.id.exit_icon)
                exitImageView.setOnClickListener {
                    alertDialog.dismiss()
                }

                alertDialog.show()
            }

        }
        button3.setOnClickListener {
            finish()
        }

        camera_btn.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(packageManager) != null) {
                val REQUEST_IMAGE_CAPTURE = 1
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }


        predict_btn.setOnClickListener {

            val classNames= listOf("Conjunctivitis Possible","Healthy!");
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            val model = LiteModel.newInstance(this)
            val inputFeature0 =
                TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            /*val predictedIndex=if(outputFeature0[0]<0.5){
                1
            }else{
                0
            }

            val predictedLabel=classNames[predictedIndex]
            resultView.text=predictedLabel*/



            var maxIdx = 0
            outputFeature0.forEachIndexed() { index, fl ->
                if (outputFeature0[maxIdx] < fl) {
                    maxIdx = index
                }
            }
            resultView.text = labels[maxIdx];
            model.close()
        }
    }
override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    if (requestCode == 100) {
        var uri = data?.data
        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
        imageView.setImageBitmap(bitmap)
    } else if (requestCode == 1) {
        try {
            bitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(bitmap)
        } catch (e: NullPointerException) {
            e.printStackTrace()
        }
    }
}
}




@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    Conjuntivitis_AppTheme {
        Greeting("Android")
    }
}