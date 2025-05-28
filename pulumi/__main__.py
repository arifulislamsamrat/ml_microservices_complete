import json
import pulumi
import pulumi_aws as aws

# Configuration
config = pulumi.Config()
vpc_cidr = "10.0.0.0/16"
public_subnet_cidr = "10.0.1.0/24"
region = aws.config.region
account_id = aws.get_caller_identity().account_id

# 1. VPC
vpc = aws.ec2.Vpc("ml-vpc",
    cidr_block=vpc_cidr,
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={"Name": "ml-vpc"}
)

# 2. Subnet
subnet = aws.ec2.Subnet("ml-subnet",
    vpc_id=vpc.id,
    cidr_block=public_subnet_cidr,
    map_public_ip_on_launch=True,
    availability_zone=f"{region}a",
    tags={"Name": "ml-subnet"}
)

# 3. Internet Gateway
igw = aws.ec2.InternetGateway("ml-igw",
    vpc_id=vpc.id,
    tags={"Name": "ml-igw"}
)

# 4. Route Table
route_table = aws.ec2.RouteTable("ml-route-table",
    vpc_id=vpc.id,
    routes=[{
        "cidr_block": "0.0.0.0/0",
        "gateway_id": igw.id
    }],
    tags={"Name": "ml-route-table"}
)

# Associate Route Table with Subnet
route_table_assoc = aws.ec2.RouteTableAssociation("ml-route-table-assoc",
    subnet_id=subnet.id,
    route_table_id=route_table.id
)

# 5. Security Group
security_group = aws.ec2.SecurityGroup("ml-sg",
    vpc_id=vpc.id,
    description="Allow HTTP traffic",
    ingress=[
        {"protocol": "tcp", "from_port": 80, "to_port": 80, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 8000, "to_port": 8000, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 8001, "to_port": 8001, "cidr_blocks": ["0.0.0.0/0"]}
    ],
    egress=[{"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}],
    tags={"Name": "ml-sg"}
)

# 6. ECS Cluster
cluster = aws.ecs.Cluster("ml-cluster", tags={"Name": "ml-cluster"})

# 7. IAM Role for ECS Task Execution
assume_role_policy = json.dumps({
    "Version": "2008-10-17",
    "Statement": [{
        "Sid": "",
        "Effect": "Allow",
        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
})

task_exec_role = aws.iam.Role("ml-task-exec-role",
    assume_role_policy=assume_role_policy,
    tags={"Name": "ml-task-exec-role"}
)

aws.iam.RolePolicyAttachment("ml-task-exec-policy",
    role=task_exec_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
)

# 8. ECR Repositories (Assuming images are already pushed)
service_a_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/service-a:latest"
service_b_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/service-b:latest"

# 9. Log Group
log_group = aws.cloudwatch.LogGroup("ml-log-group",
    retention_in_days=7,
    tags={"Name": "ml-log-group"}
)

# 10. Task Definitions
def create_task_definition(name, image, container_port):
    def build_container_def(log_group_name):
        container_def = [{
            "name": name,
            "image": image,
            "portMappings": [{
                "containerPort": container_port,
                "hostPort": container_port,
                "protocol": "tcp"
            }],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": str(log_group_name),
                    "awslogs-region": region,
                    "awslogs-stream-prefix": name
                }
            },
            "essential": True
        }]
        return json.dumps(container_def)

    return aws.ecs.TaskDefinition(f"{name}-task-def",
        family=name,
        cpu="256",
        memory="512",
        network_mode="awsvpc",
        requires_compatibilities=["FARGATE"],
        execution_role_arn=task_exec_role.arn,
        container_definitions=log_group.name.apply(build_container_def)
    )



task_def_a = create_task_definition("service-a", service_a_image, 8000)
task_def_b = create_task_definition("service-b", service_b_image, 8001)

# 11. Load Balancer
alb = aws.lb.LoadBalancer("ml-alb",
    security_groups=[security_group.id],
    subnets=[subnet.id],
    tags={"Name": "ml-alb"}
)

target_group_a = aws.lb.TargetGroup("tg-service-a",
    port=8000,
    protocol="HTTP",
    target_type="ip",
    vpc_id=vpc.id,
    health_check={
        "path": "/health",
        "protocol": "HTTP",
        "port": "8000"
    },
    tags={"Name": "tg-service-a"}
)

target_group_b = aws.lb.TargetGroup("tg-service-b",
    port=8001,
    protocol="HTTP",
    target_type="ip",
    vpc_id=vpc.id,
    health_check={
        "path": "/health",
        "protocol": "HTTP",
        "port": "8001"
    },
    tags={"Name": "tg-service-b"}
)

listener = aws.lb.Listener("ml-listener",
    load_balancer_arn=alb.arn,
    port=80,
    default_actions=[{
        "type": "forward",
        "target_group_arn": target_group_a.arn
    }],
    tags={"Name": "ml-listener"}
)

# 12. ECS Services
def create_service(name, task_def, target_group, container_port):
    return aws.ecs.Service(f"{name}-service",
        cluster=cluster.arn,
        desired_count=1,
        launch_type="FARGATE",
        task_definition=task_def.arn,
        network_configuration={
            "subnets": [subnet.id],
            "security_groups": [security_group.id],
            "assign_public_ip": True
        },
        load_balancers=[{
            "target_group_arn": target_group.arn,
            "container_name": name,
            "container_port": container_port
        }],
        opts=pulumi.ResourceOptions(depends_on=[listener]),
        tags={"Name": f"{name}-service"}
    )

service_a = create_service("service-a", task_def_a, target_group_a, 8000)
service_b = create_service("service-b", task_def_b, target_group_b, 8001)

# 13. Outputs
pulumi.export("alb_dns_name", alb.dns_name)
pulumi.export("service_a_url", alb.dns_name.apply(lambda dns: f"http://{dns}/"))
pulumi.export("service_b_url", alb.dns_name.apply(lambda dns: f"http://{dns}/"))
