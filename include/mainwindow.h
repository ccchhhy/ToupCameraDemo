#pragma once

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer.h>
#include "appinit.h"
#include "appevent.h"
#include "dialog.h"
#include "ImageProcessor.h"
#include <vector>
#include <QtConcurrent>
#include "mythread.h"
#include "connx.h"
#include "devicemanager.h"
#include "form.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE



class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    AppInit m_appInit;
    AppEvent &m_appEvent = AppEvent::instance();
    // 获取DeviceManager类的唯一实例
    DeviceManager& m_deviceManager = DeviceManager::getInstance();
    CCamera* m_camera = nullptr;
    CWebSocketServer *m_webSocketServer = nullptr;
    QJsonObject m_jsonObj;
    COnnx* onnx = nullptr;
    MyThread *thread = new MyThread(this);
    ImageProcessor processor;

    void setupUI();
    void setDefaultValues();
    void setDefaultStates();
    void registerEvents();
    void unregisterEvents();
    void signalSlotConnect();
    void getCameraList();
    void setCameraUI();

    int m_camIndex = 0;
    std::string m_camType = "USB";
    QTimer* m_readTimer;

    int m_fps = 30;
    int m_port = 10086;
    bool m_isGPU = false;
    std::string model_name = "NAF";

private slots:
    void handleEvent(const QString& eventName, const QVariant& eventData);
    void showCameraFrame();

     void on_m_btn_open_camera_clicked();
     void on_m_cbx_camera_list_currentIndexChanged(int index);
     void on_m_cbx_camera_type_currentIndexChanged(int index);
     void on_m_cbx_camera_type_currentTextChanged(const QString &arg1);
     void on_action_normal_triggered();
     void on_action_maxscreen_triggered();
     void on_action_fullscreen_triggered();
     void on_action_exit_triggered();
     void on_action_open_image_triggered();
     void on_action_about_triggered();

     void on_m_btn_open_web_socket_clicked();
     void on_m_line_port_textChanged(const QString &arg1);
     void onWebSocketTextMessageReceived(QWebSocket *clientSocket, const QString& message);
     void onWebSocketBinaryMessageReceived(QWebSocket *clientSocket, const QByteArray& message);
     void onWebSocketClientConnected(QWebSocket *clientSocket);
     void onWebSocketClientDisconnected(QWebSocket *clientSocket);
     void on_comboBox_3_currentTextChanged(const QString &arg1);
     void on_pushButton_2_clicked();
     void on_checkBox_4_stateChanged(int arg1);
     void on_comboBox_2_currentTextChanged(const QString &arg1);
     void on_m_btn_load_algorithm_clicked(bool checked);
     void on_m_btn_send_message_clicked();
     void on_horizontalSlider_8_valueChanged(int value);
     void on_expTargetlSlider_valueChanged(int value);
     void on_expTimeSlider_valueChanged(int value);
     void on_expGainSlider_valueChanged(int value);
     void on_autoExpocheckBox_stateChanged(int arg1);
     void on_btn_open_dialog_clicked();
     void on_m_cmb_resolution_currentIndexChanged(int index);
};
#endif // MAINWINDOW_H
