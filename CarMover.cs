using System.Collections;
using UnityEngine;

public class CarMover : MonoBehaviour
{
    private TargetWaypoint _targetWaypoint;
    private float distanceToWaypoint = 0.1f;

    private Transform currentWaypoint;
    public Vector3 CurrentPosition { get; private set; }

    private float moveSpeed; // 현재 이동 속도

    private void Awake()
    {
        // TargetWaypoint를 찾고 초기 설정
        _targetWaypoint = FindObjectOfType<TargetWaypoint>();

        // 첫 번째 웨이포인트 설정 및 초기 위치 이동
        currentWaypoint = _targetWaypoint.GetNextWayPoint(null);
        transform.position = currentWaypoint.position;

        // 다음 웨이포인트로 설정 및 방향 전환
        currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);
        transform.LookAt(currentWaypoint);

        // 초기 속도를 현재 웨이포인트 속도로 설정
        moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);
    }

    private void FixedUpdate()
    {
        if (currentWaypoint == null)
        {
            // 현재 웨이포인트가 null이면 첫 번째 웨이포인트로 설정
            currentWaypoint = _targetWaypoint.GetNextWayPoint(null);
            transform.LookAt(currentWaypoint);
        }

        // 웨이포인트로 이동
        transform.position = Vector3.MoveTowards(transform.position, currentWaypoint.position, moveSpeed * Time.deltaTime);

        // 웨이포인트에 도달했는지 확인
        if (Vector3.Distance(transform.position, currentWaypoint.position) < distanceToWaypoint)
        {
            // 다음 웨이포인트로 전환
            currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);

            if (currentWaypoint != null)
            {
                // 다음 웨이포인트의 속도를 가져와 이동 속도 업데이트
                moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);

                // 다음 웨이포인트로 방향 전환
                transform.LookAt(currentWaypoint);
            }
        }
    }
}
