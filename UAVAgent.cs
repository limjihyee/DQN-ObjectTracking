using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.UI;

public class UAVAgent : Agent
{
    private Transform tr; // uav
    private Rigidbody rb;
    private RayPerceptionSensorComponent3D raySensorComponent;
    private GameObject detectorCameraObject;
    private Camera detectorCamera;
    private RenderTexture targetTexture;
    private Texture2D carTexture;
    private Text TextInfo;

    private RaycastHit hitobsFront;

    //private Vector3 leftDiagonal = Quaternion.Euler(0, -5, 0) * Vector3.forward;
    //private Vector3 rightDiagonal = Quaternion.Euler(0, 5, 0) * Vector3.forward;

    private float reward = 0.0f;
    private float speedreward = 0.0f;

    private Vector3 startPositionWorld;
    private Vector3 endPositionWorld;
    private Vector3 hitPosition;

    private Vector3 startPosition;
    private Vector3 startRotation;
    //private const float disObs = 30.0f;
    private static float horizontalSpeed = 3.0f; //Initial에서 초기화
    private static float verticalSpeed = 4.0f;
    private readonly float maxHorizontalSpeed = 4.0f;
    private float[] hSpeedObs = new float[4] { 3.0f, 3.0f, 3.0f, 3.0f }; // 속도 input

    private static float altitude = 26.0f;

    private Vector3 horizontalDirection;
    private Vector3 verticalDirection;
    private Vector3 movementDirection;

    private readonly Vector3 m_Forward = Vector3.forward;
    private readonly Vector3 m_Back = Vector3.back;
    private readonly Vector3 m_Left = Vector3.left;
    private readonly Vector3 m_Right = Vector3.right;
    private readonly Vector3 m_ForwardLeft = (Vector3.forward + Vector3.left).normalized;
    private readonly Vector3 m_ForwardRight = (Vector3.forward + Vector3.right).normalized;
    private readonly Vector3 m_BackLeft = (Vector3.back + Vector3.left).normalized;
    private readonly Vector3 m_BackRight = (Vector3.back + Vector3.right).normalized;
    private readonly Vector3 m_Up = Vector3.up * verticalSpeed;
    private readonly Vector3 m_Down = Vector3.down * verticalSpeed;

    //private float disLeft = 0f;
    //private float disRight = 0f;
    private Color[] pixels;

    private GameObject carMover;
    private Vector3 carPosition;
    //private float distance = 0.0f;

    private float[] positionObs = new float[9] { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f }; // UAV 위치 2 차량 위치 2*4 및 차량 위치 토글 1 - > 원래 11인데 통합하니까
    private Vector3 previousCarPosition;  // 차량의 이전 위치를 저장할 변수
    private float distanceToVehicle; 
    private float angleToVehicle;
    private float dx;
    private float dz;
    private float distanceXZ;

    private float zoomSpeed = 2.0f; // 줌 속도 설정
    private float minFOV = 30f; // 최소 FOV (줌 인 상태)
    private float maxFOV = 90f; // 최대 FOV (줌 아웃 상태)

    //todo text info
    private float totalReward = 0.0f;
    private int episode = 0;
    private int step = 0;
    private bool wasVehicleDetected = false;  // 이전 스텝에서 차량이 감지되었는지 여부


    public override void Initialize()
    {
        // Time.timeScale = 2.0f;  // 시간 스케일
        Time.fixedDeltaTime = 0.05f;  // 1초에 20번 step 실행

        tr = GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();

        startPosition = tr.position;
        startRotation = tr.eulerAngles;

        detectorCameraObject = GameObject.Find("DetectCamera");

        detectorCamera = detectorCameraObject.GetComponent<Camera>();
        detectorCamera.targetTexture = new RenderTexture(16, 16, 16);
        targetTexture = detectorCamera.targetTexture;

        carMover = GameObject.Find("targetVehicle");
        TextInfo = GameObject.Find("TextInfo").GetComponent<Text>();
        TextInfo.text = "Start///";
    }

    public override void OnEpisodeBegin()
    {
        // UAV의 속도 초기화
        rb.angularVelocity = Vector3.zero;
        horizontalSpeed = 3.0f;
        altitude = 26.0f;

        // 차량과 UAV의 위치를 동일하게 초기화
        tr.position = new Vector3(carMover.transform.position.x, 26f, carMover.transform.position.z); // 차량의 위치를 가져와서 UAV 위치로 설정
        tr.eulerAngles = startRotation; // UAV의 초기 회전값 유지

        // UAV와 차량의 시작 위치 정보 설정
        positionObs[0] = Mathf.Round(tr.position.x * 100f) / 100f;
        positionObs[1] = Mathf.Round(tr.position.z * 100f) / 100f;
        //positionObs[2] = Mathf.Round(carMover.transform.position.x * 100f) / 100f;
        //positionObs[3] = Mathf.Round(carMover.transform.position.z * 100f) / 100f;
        positionObs[2] = 1f;

        // 카메라 텍스처 및 리워드 초기화
        carTexture = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB565, false);
        SetReward(0);

        // UAV의 방향 및 스텝 초기화 (action)
        horizontalDirection = Vector3.zero;
        verticalDirection = Vector3.zero;
        step = 0;
        episode++;

        //speed리워드 한 에피소드당 얼마나 찍히는지 보려고
        speedreward = 0.0f;
    }

    //
    // //이미지 저장 함수 
    // private void SaveImageToDisk(Texture2D texture, string filePath)
    // {
    //     byte[] bytes = texture.EncodeToPNG();
    //     System.IO.File.WriteAllBytes(filePath, bytes);
    //     Debug.Log($"Image saved to {filePath}");
    // }

    private bool detectVehicle()
    {
        // 카메라 설정 및 렌더링
        targetTexture = detectorCamera.targetTexture;
        RenderTexture.active = targetTexture;
        detectorCamera.Render();

        // 텍스처에서 이미지를 읽음
        carTexture.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
        carTexture.Apply();
        pixels = carTexture.GetPixels();

        // // DQN에 들어가는 이미지 확인을 위해 이미지 저장
        // SaveImage(carTexture);  // 디버깅용 이미지 저장 함수 호출

        // 픽셀 분석 (detectcamera 상 픽셀 색 있으면 detect로 판단)
        foreach (Color pixel in pixels)
        {
            // Debug.Log($"Pixel Color - R: {pixel.r}, G: {pixel.g}, B: {pixel.b}");

            if (pixel.r > 0.01f || pixel.g > 0.01f || pixel.b > 0.01f)
            {
                carPosition = carMover.transform.position;
                return true;
            }
        }
        carPosition = Vector3.zero;
        return false;
    }

    public override void CollectObservations(VectorSensor sensor) // 15
    {
        for (var i = 0; i < 9; i++)
        {
            sensor.AddObservation(positionObs[i]);
        }
        sensor.AddObservation(distanceToVehicle);
        sensor.AddObservation(angleToVehicle);
        for (var i = 0; i < 4; i++)
        {
            sensor.AddObservation(hSpeedObs[i]);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var action = actions.DiscreteActions[0];
        reward = 0.0f;
        SetReward(0);
        step++;

        //todo action
        switch (action)
        {
            case 0: // 정지
                horizontalDirection = Vector3.zero;
                break;
            case 1: // 앞으로 이동
                horizontalDirection = m_Forward;
                verticalDirection = Vector3.zero;
                break;
            case 2: // 뒤로 이동
                horizontalDirection = m_Back;
                verticalDirection = Vector3.zero;
                break;
            case 3: // 왼쪽 이동
                horizontalDirection = m_Left;
                verticalDirection = Vector3.zero;
                break;
            case 4: // 오른쪽 이동
                horizontalDirection = m_Right;
                verticalDirection = Vector3.zero;
                break;
            case 5: // 왼쪽 대각선 앞으로 이동
                horizontalDirection = m_ForwardLeft;
                verticalDirection = Vector3.zero;
                break;
            case 6: // 오른쪽 대각선 앞으로 이동
                horizontalDirection = m_ForwardRight;
                verticalDirection = Vector3.zero;
                break;
            case 7: // 왼쪽 대각선 뒤로 이동
                horizontalDirection = m_BackLeft;
                verticalDirection = Vector3.zero;
                break;
            case 8: // 오른쪽 대각선 뒤로 이동
                horizontalDirection = m_BackRight;
                verticalDirection = Vector3.zero;
                break;
            case 9: // 가속
                if (horizontalSpeed < maxHorizontalSpeed) // 최대 속도 제한
                {
                    horizontalSpeed += 0.3f;
                    horizontalSpeed = Mathf.Round(horizontalSpeed * 10f) / 10f;
                }
                verticalDirection = Vector3.zero;
                break;
            case 10: // 감속
                if (horizontalSpeed > 0)
                {
                    horizontalSpeed -= 0.3f;
                    horizontalSpeed = Mathf.Round(horizontalSpeed * 10f) / 10f;
                }
                verticalDirection = Vector3.zero;
                break;
            case 11: // 고도 하강
                if (altitude > 0)
                {
                    verticalDirection = m_Down;
                }
                break;
            case 12: // 고도 상승
                verticalDirection = m_Up;
                break;
            case 13: // 줌 인
                detectorCamera.fieldOfView = Mathf.Clamp(detectorCamera.fieldOfView - zoomSpeed, minFOV, maxFOV);
                break;
            case 14: // 줌 아웃
                detectorCamera.fieldOfView = Mathf.Clamp(detectorCamera.fieldOfView + zoomSpeed, minFOV, maxFOV);
                break;
            default: // 오류 처리
                horizontalDirection = Vector3.zero;
                verticalDirection = Vector3.zero;
                break;
        }

        
        // UAV 이동 처리
        movementDirection = horizontalDirection * horizontalSpeed + verticalDirection;
        transform.Translate(movementDirection * Time.deltaTime);

        // todo reward
        // 차량 포착에 따른 reward
        bool isVehicleDetected = detectVehicle();

        if (isVehicleDetected)// && wasVehicleDetected)
        {
            reward += 1.0f;  // 차량이 계속 포착되고 있는 경우
        }
        // else if (isVehicleDetected && !wasVehicleDetected)
        // {
        //     reward += 10.0f;  // 차량이 새롭게 포착된 경우
        // }
        else if (!isVehicleDetected && wasVehicleDetected)
        {
            reward += -10.0f;  // 차량이 사라진 경우
        }
        else
        {
            reward += -1.0f;  // 차량이 계속 포착되지 않는 경우
        }
        
        // 차량과 uav 거리 기반 reward 및 수평속도 reward
        if (isVehicleDetected)
        {
            // 차량과 uav 거리 기반 reward
            carPosition = carMover.transform.position;
            //float distanceToCar = Vector3.Distance(tr.position, carPosition);

            // y좌표가 거리 25를 유지하게끔 변경
            Vector3 trY = new Vector3(0, tr.position.y, 0);
            Vector3 carY = new Vector3(0, carPosition.y, 0);
            float distanceToCar = Vector3.Distance(trY, carY);

            // 거리 기반 선형 보상
            float maxDistance = 50.0f; //최대거리
            float optimalDistance = 25.0f;  // 최적 거리
            float distanceDifference = Mathf.Abs(distanceToCar - optimalDistance); //얼마나 벗어났는지
            float rewardScalingFactor = 1.0f;  // 보상 스케일링을 위한 계수

            // 거리가 가까울수록 큰 보상을 주고, 멀어질수록 선형적으로 감소
            reward += Mathf.Round(rewardScalingFactor * (1.0f - (distanceDifference / maxDistance)) * 100f) / 100f;

            // 속도 reward
            dx = carPosition.x - tr.position.x;
            dz = carPosition.z - tr.position.z;
            distanceXZ = Mathf.Sqrt(Mathf.Pow(dx, 2) + Mathf.Pow(dz, 2));

            // 수평 속도 보상 조건 설정
            if (distanceXZ >= 0 && distanceXZ <= 3)
            {
                if (action != 9 && action != 10) // 가/감속하지 않을 때
                {
                    reward += 0.5f;
                    speedreward += 0.5f;
                }
            }
            else if (distanceXZ > 3 && distanceXZ <= 15)
            {
                if (action == 9 && !Mathf.Approximately(horizontalSpeed, maxHorizontalSpeed)) // 가속할 때
                {
                    reward += 1.0f;
                    speedreward += 1.0f;
                }
            }
        }
        else
        {
            // 차량이 감지되지 않았고 가속을 할 때 리워드
            if (action == 9)
            {
                reward += 0.5f;
                speedreward += 0.5f;
            }
        }

        wasVehicleDetected = isVehicleDetected;
        
        // UAV의 위치 및 보상 업데이트
        SetReward(reward);
        totalReward = GetCumulativeReward();
        
        // observation 값 넘기는 부분 (CollectObservations)
        for (var i = 0; i < 6; i++) // uav 위치
        {
            positionObs[i] = positionObs[i + 2];
        }
        positionObs[6] = Mathf.Round(tr.position.x * 100f) / 100f;
        positionObs[7] = Mathf.Round(tr.position.z * 100f) / 100f;

        if (isVehicleDetected)
        {
            //positionObs[8] = Mathf.Round(carPosition.x * 100f) / 100f; //차량
            //positionObs[9] = Mathf.Round(carPosition.z * 100f) / 100f;
            positionObs[8] = 1f; // 토글
            // 2. uav와 차량 사이의 거리 관찰값
            distanceToVehicle = Mathf.Round(Vector3.Distance(tr.position, carPosition) * 100f) / 100f;
            // 3. uav와 차량까지의 방향 벡터로 회전할 각도 관찰값
            dx = carPosition.x - tr.position.x;  // UAV와 차량 사이의 x축 차이
            dz = carPosition.z - tr.position.z;  // UAV와 차량 사이의 z축 차이
            // Atan2로 각도를 구하고, 라디안 값을 각도로 변환
            angleToVehicle = Mathf.Atan2(dx, dz) * Mathf.Rad2Deg;
            SendCarPositionToPython(carPosition); // python으로 전송
        }
        else
        {
            positionObs[8] = 0f;
            //positionObs[9] = 0f;
            //positionObs[10] = 0f;
            distanceToVehicle = 0.0f;
            angleToVehicle = 0.0f;
            distanceXZ = 0.0f;
        }

        for (var i = 0; i < 3; i++) // 수평 속도 관찰값
        {
            hSpeedObs[i] = hSpeedObs[i + 1];
        }
        hSpeedObs[3] = horizontalSpeed;

        //todo text info
        var info = $"Episode: {episode / 2}\n" + // 수정 필요
                   $"Step: {step}\n" +
                   $"Action: {action}\n" +
                   $"HSpeed: {horizontalSpeed}\n" +
                   $"Detect: {isVehicleDetected}\n" +
                   $"Reward: {reward}\n" +
                   //$"speedReward: {speedreward}\n" +
                   //$"Score: {totalReward}\n" +
                   $"UAVxzy: {positionObs[0]}, {positionObs[1]}, {tr.position.y}\n" +
                   //$"CARxz: {positionObs[8]}, {positionObs[9]}\n" +
                   $"CARxz: {carPosition.x}, {carPosition.z}\n" +
                   $"xyzdistance: {distanceToVehicle}\n" +
                   //$"xzdistance: {distanceXZ}\n" +
                   $"angle: {angleToVehicle}\n" +
                   $"FOV: {detectorCamera.fieldOfView}\n"+
                   "";

        TextInfo.text = info;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actionOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.W)) actionOut[0] = 1;
        if (Input.GetKey(KeyCode.S)) actionOut[0] = 2;
        if (Input.GetKey(KeyCode.A)) actionOut[0] = 3;
        if (Input.GetKey(KeyCode.D)) actionOut[0] = 4;
        if (Input.GetKey(KeyCode.Q)) actionOut[0] = 0;
        if (Input.GetKey(KeyCode.R)) actionOut[0] = 10;
        if (Input.GetKey(KeyCode.V)) actionOut[0] = 11;
        if (Input.GetKey(KeyCode.F)) actionOut[0] = 9;
        if (Input.GetKey(KeyCode.Z)) actionOut[0] = 13; // 줌 인
        if (Input.GetKey(KeyCode.X)) actionOut[0] = 14; // 줌 아웃
    }
    // 두 개의 물체가 서로 충돌한 순간에 호출 (물리적인 충돌에 반응, Rigidbody가 있는 물체끼리 충돌할 때)
    private void OnCollisionEnter(Collision collision)
    {
        // 지형지물 충돌 시
        if (collision.collider.CompareTag("AbstractMap"))
        {
            AddReward(-200.0f);
            //Debug.Log("AbstractMap");
            EndEpisode();
            //System.GC.Collect(); //가비지 컬렉션(GC) 수행.
        }

        // if (collision.collider.CompareTag("Tunnel"))
        // {
        //     AddReward(-200.0f); // 충돌 페널티
        //     EndEpisode(); // 에피소드 종료
        // }
        
    }

    private void SendCarPositionToPython(Vector3 position)
    {
        try
        {
            using (System.Net.Sockets.TcpClient client = new System.Net.Sockets.TcpClient("127.0.0.1", 9999))
            {
                using (System.IO.StreamWriter writer = new System.IO.StreamWriter(client.GetStream()))
                {
                    string message = $"{position.x},{position.y},{position.z}";
                    writer.WriteLine(message);
                    writer.Flush();
                }
            }
        }
        catch (Exception e)
        {
            Debug.Log("Socket send error: " + e.Message);
        }
    }
}