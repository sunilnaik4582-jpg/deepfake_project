@echo off
echo ============================================================
echo  Zipping dataset for Google Colab upload...
echo ============================================================
cd /d "%~dp0"
powershell -Command "Compress-Archive -Path 'dataset' -DestinationPath 'dataset.zip' -Force"
echo.
echo ✅ Done! dataset.zip created.
echo.
echo NEXT STEPS:
echo  1. Upload dataset.zip to Google Drive (My Drive root)
echo  2. Open deepfake_colab_train.ipynb in Google Colab
echo  3. Go to Runtime → Change Runtime Type → GPU (T4)
echo  4. Run all cells in order
echo  5. Download deepfake_model.h5 at the end
echo  6. Replace model\deepfake_model.h5 with the new file
echo.
pause
