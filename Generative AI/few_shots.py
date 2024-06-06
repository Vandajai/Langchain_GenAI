few_shots = [
    {"input": "List all accounts.", "query": "SELECT * FROM [account];"},
    {
        "input": "Count account ID of account name adm?",
        "query": "SELECT COUNT(ID) FROM [account] WHERE acct_name ='bin';",
    },
    {
        "input": "List all group having platform name 'Linux Server'",
        "query": "SELECT g.* FROM [group] g JOIN [membership] m ON g.id = m.m_group_id JOIN [account] a ON a.id = m.account_id WHERE a.platform_name='Linux Server';",
    },
    {
        "input": "Consider display name as user name and provide all the user name where group name is Parisdev'",
         "query": "SELECT a.display_name AS user_name FROM [account] a INNER JOIN [membership] m ON a.id = m.account_id INNER JOIN [group] g ON m.m_group_id = g.id WHERE g.group_name = 'Parisdev';",
    },
    {
        "input": "Provide group name for account name root",
         "query": "SELECT g.group_name FROM [group] g INNER JOIN [membership] m ON g.id = m.m_group_id INNER JOIN [account] a ON m.account_id = a.id WHERE a.acct_name = 'root';",
    },
]
