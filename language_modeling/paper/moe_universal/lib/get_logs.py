from wandb_gql import gql

def get_logs(run):
    QUERY = gql('''
    query RunLogLines($projectName: String!, $entityName: String, $runName: String!) {
        project(name: $projectName, entityName: $entityName) {
            id
            run(name: $runName) {
                id
                logLines(first: 1000000) {
                    edges {
                        node {
                            line
                            }
                    }
                }
            }
        }
    }
    ''')


    response = run.client.execute(QUERY, variable_values={
        'entityName': run.entity,
        'projectName': run.project,
        'runName': run.id,
    })

    logs = []
    for l in response["project"]["run"]["logLines"]["edges"]:
        line = l["node"]["line"]
        logs.append(line.strip())
    return logs
