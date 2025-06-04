using System.Collections;
using UnityEngine;

public class CarMover : MonoBehaviour
{
    private TargetWaypoint _targetWaypoint;
    private float distanceToWaypoint = 0.1f;

    private Transform currentWaypoint;
    public Vector3 CurrentPosition { get; private set; }

    private float moveSpeed; // ���� �̵� �ӵ�

    private void Awake()
    {
        // TargetWaypoint�� ã�� �ʱ� ����
        _targetWaypoint = FindObjectOfType<TargetWaypoint>();

        // ù ��° ��������Ʈ ���� �� �ʱ� ��ġ �̵�
        currentWaypoint = _targetWaypoint.GetNextWayPoint(null);
        transform.position = currentWaypoint.position;

        // ���� ��������Ʈ�� ���� �� ���� ��ȯ
        currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);
        transform.LookAt(currentWaypoint);

        // �ʱ� �ӵ��� ���� ��������Ʈ �ӵ��� ����
        moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);
    }

    private void FixedUpdate()
    {
        if (currentWaypoint == null)
        {
            // ���� ��������Ʈ�� null�̸� ù ��° ��������Ʈ�� ����
            currentWaypoint = _targetWaypoint.GetNextWayPoint(null);
            transform.LookAt(currentWaypoint);
        }

        // ��������Ʈ�� �̵�
        transform.position = Vector3.MoveTowards(transform.position, currentWaypoint.position, moveSpeed * Time.deltaTime);

        // ��������Ʈ�� �����ߴ��� Ȯ��
        if (Vector3.Distance(transform.position, currentWaypoint.position) < distanceToWaypoint)
        {
            // ���� ��������Ʈ�� ��ȯ
            currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);

            if (currentWaypoint != null)
            {
                // ���� ��������Ʈ�� �ӵ��� ������ �̵� �ӵ� ������Ʈ
                moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);

                // ���� ��������Ʈ�� ���� ��ȯ
                transform.LookAt(currentWaypoint);
            }
        }
    }
}
